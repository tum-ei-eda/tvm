/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file relay/backend/graph_plan_memory.cc
 * \brief Memory index assignment pass for executing
 *   the program in the graph executor.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../analysis/dependency_graph.h"

namespace tvm {
namespace relay {

using IntegerArray = Array<Integer>;

struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type node. */
  const TensorTypeNode* ttype{nullptr};
  /*! \brief virtual device index that corresponds to the device_type in
   * DLDevice. */
  int device_type{0};
  /*! \brief The storage id */
  int64_t storage_id{-1};
};

class StorageAllocaBaseVisitor : public ExprVisitor {
 public:
  // run the visitor on a function.
  void Run(const Function& func) {
    for (Var param : func->params) {
      CreateToken(param.operator->(), false);
    }
    // must always keep output alive.
    for (StorageToken* tok : GetToken(func->body)) {
      tok->ref_counter += 1;
    }
  }

  void VisitExpr_(const ConstantNode* op) final { this->CreateToken(op, false); }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<StorageToken*> fields;
    for (Expr field : op->fields) {
      auto tokens = GetToken(field);
      fields.insert(fields.end(), tokens.begin(), tokens.end());
    }
    token_map_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& tok = GetToken(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), tok.size());
    token_map_[op] = {tok[op->index]};
  }

  void VisitExpr_(const IfNode* op) final { LOG(FATAL) << "if is not supported."; }

  void VisitExpr_(const LetNode* op) final {
    auto token = GetToken(op->value);
    token_map_[op->var.operator->()] = token;
    token_map_[op] = GetToken(op->body);
  }

 protected:
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > token_map_;

  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<StorageToken*>& GetToken(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = token_map_.find(expr.operator->());
    ICHECK(it != token_map_.end());
    return it->second;
  }
  /*!
   * \brief Populate the token map to set op's tokens
   * \param op The node to be processed.
   * \param can_realloc Whether we can re-allocate the memory.
   */
  virtual void CreateToken(const ExprNode* op, bool can_realloc) = 0;
};

class StorageAllocaInit : protected StorageAllocaBaseVisitor {
 public:
  explicit StorageAllocaInit(support::Arena* arena) : arena_(arena) {}

  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > GetInitTokenMap(
      const Function& func) {
    node_device_map_ = CollectDeviceInfo(func);
    this->Run(func);
    return std::move(token_map_);
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;

  void CreateToken(const ExprNode* op, bool can_realloc) final {
    ICHECK(!token_map_.count(op));
    std::vector<StorageToken*> tokens;
    int device_type =
        node_device_map_.count(GetRef<Expr>(op)) ? node_device_map_[GetRef<Expr>(op)]->value : 0;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        ICHECK(ttype);
        StorageToken* token = arena_->make<StorageToken>();
        token->ttype = ttype;
        token->device_type = device_type;
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      ICHECK(ttype);
      StorageToken* token = arena_->make<StorageToken>();
      token->ttype = ttype;
      token->device_type = device_type;
      tokens.push_back(token);
    }
    token_map_[op] = tokens;
  }

  void VisitExpr_(const CallNode* op) final {
    // create token for the call node.
    CreateToken(op, true);
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        tok->ref_counter += 1;
      }
    }
  }

 private:
  // allocator
  support::Arena* arena_;
  Map<Expr, Integer> node_device_map_;
};

class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  /*!
   * \return totoal number of bytes allocated
   */
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (const auto* p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // Run storage allocation for a function.
  Map<Expr, Array<IntegerArray> > Plan(const Function& func) {
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);
    this->Run(func);

    // The value of smap contains two integer arrays where the first array
    // contains the planned storage ids and the second holds the device types.
    Map<Expr, Array<IntegerArray> > smap;
    int num_annotated_nodes = 0;
    int num_nodes = 0;

    for (const auto& kv : token_map_) {
      std::vector<Integer> storage_ids;
      std::vector<Integer> device_types;
      std::vector<Integer> sid_sizes_byte;
      for (StorageToken* tok : kv.second) {
        if (tok->device_type) {
          num_annotated_nodes++;
        }
        num_nodes++;
        storage_ids.push_back(tok->storage_id);
        device_types.push_back(tok->device_type);
        sid_sizes_byte.push_back(GetMemorySize(tok));
      }
      smap.Set(GetRef<Expr>(kv.first),
               Array<IntegerArray>({storage_ids, device_types, sid_sizes_byte}));
    }
    // Either all or none of the nodes should be annotated.
    if (num_annotated_nodes != 0 && num_annotated_nodes != num_nodes) {
      LOG(FATAL) << num_annotated_nodes << " out of " << num_nodes
                 << "expressions are assigned with virtual device types. Either all "
                    "or none of the expressions are expected to be annotated.";
    }
    return smap;
  }

 protected:
  using StorageAllocaBaseVisitor::VisitExpr_;
  // override create token by getting token as prototype requirements.
  void CreateToken(const ExprNode* op, bool can_realloc) final {
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    std::vector<StorageToken*> tokens;

    for (StorageToken* tok : it->second) {
      if (can_realloc) {
        tokens.push_back(Request(tok));
      } else {
        // Allocate a new token,
        StorageToken* allocated_tok = Alloc(tok, GetMemorySize(tok));
        allocated_tok->device_type = tok->device_type;
        // ensure it never get de-allocated.
        allocated_tok->ref_counter += 1;
        tokens.push_back(allocated_tok);
      }
    }
    token_map_[op] = tokens;
  }
  // Mark op to reuse the input_token
  // tie the two memories together
  void ReuseInputToken(const ExprNode* op, StorageToken* input_token) {
    ICHECK(!token_map_.count(op));
    auto it = prototype_.find(op);
    ICHECK(it != prototype_.end());
    ICHECK_EQ(it->second.size(), 1U);
    StorageToken* prototype = it->second[0];
    // add the reference counter of the output
    // so the input token can only be deleted after references
    // to both are expired
    input_token->ref_counter += prototype->ref_counter;
    // reuse the input token
    token_map_[op] = {input_token};
  }

  // The call map
  void VisitExpr_(const CallNode* op) final {
    std::vector<StorageToken*> args;
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (StorageToken* tok : GetToken(arg)) {
        args.push_back(tok);
      }
    }
    // Under the flat-memory setting.
    // we can force aliasing the input and output of reshape
    // to make it an nop. Note that this is not true
    // for non-flat memory case. Given the current graph plan memory
    // only works for flat memory case, we will go with this choice
    //
    // TODO(tvm-team) Update checks of flat memory enablement when we support
    // opaque-nd memory planning to skip this path.
    if (IsReshape(op)) {
      ICHECK_EQ(args.size(), 1U);
      ReuseInputToken(op, args[0]);
    } else {
      // create token for the call node.
      CreateToken(op, true);
    }
    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : token_map_.at(op)) {
      CheckForRelease(tok);
    }
    for (StorageToken* tok : args) {
      tok->ref_counter -= 1;
      CheckForRelease(tok);
    }
  }
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static size_t DivRoundUp(size_t size, size_t word_size) {
    return (size + word_size - 1) / word_size;
  }
  /*!
   * \brief The call is an reshape only op
   * \param call The call to be checked.
   * \return the check result.
   */
  static bool IsReshape(const CallNode* call) {
    if (const auto* fn = call->op.as<FunctionNode>()) {
      return fn->HasNonzeroAttr(attr::kReshapeOnly);
    }
    return false;
  }
  /*!
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  size_t GetMemorySize(StorageToken* prototype) {
    const TensorTypeNode* ttype = prototype->ttype;
    ICHECK(ttype != nullptr);
    size_t size = 1;
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ICHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
      ICHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
      size *= static_cast<size_t>(pval[0]);
    }
    size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
    return size;
  }
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype) {
    // calculate the size;
    size_t size = GetMemorySize(prototype);
    // search memory block in [size / match_range_, size * match_range_)
    if (match_range_ == 0) {
      return this->Alloc(prototype, size);
    }
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      free_.erase(it);
      return tok;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken* tok = it->second;
      if (tok->device_type != prototype->device_type) continue;
      ICHECK_EQ(tok->ref_counter, 0);
      // Use exect matching strategy
      tok->max_bytes = std::max(size, tok->max_bytes);
      tok->ref_counter = prototype->ref_counter;
      // erase from map and return
      free_.erase(it);
      return tok;
    }
    // cannot find anything return a new one.
    return this->Alloc(prototype, size);
  }
  /*!
   * \brief Allocate a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, size_t size) {
    prototype->max_bytes = size;
    prototype->storage_id = static_cast<int64_t>(data_.size());
    data_.push_back(prototype);
    return prototype;
  }
  /*!
   * \brief Check if we can release token.
   * \param tok The token to be released.
   */
  void CheckForRelease(StorageToken* tok) {
    ICHECK_GE(tok->storage_id, 0);
    ICHECK_GE(tok->ref_counter, 0);
    if (tok->ref_counter == 0) {
      free_.insert({tok->max_bytes, tok});
    }
  }

 private:
  // allocator
  support::Arena arena_;
  // scale used for rough match
  size_t match_range_{16};
  // free list of storage entry
  std::multimap<size_t, StorageToken*> free_;
  // all the storage resources available
  std::vector<StorageToken*> data_;
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > prototype_;
};

class DbgVisitor : public ExprVisitor {
  public:
  std::string GetSimpleExprName(const Expr &expr) {
    VisitExpr(expr);
    return m_dbgStr;
  }
  std::string GetSimpleExprName(const ExprNode *expr) {
    return GetSimpleExprName(GetRef<Expr>(expr));
  }

  void VisitExpr_(const VarNode *var) final {
    m_dbgStr += "Var_name(" + var->name_hint() + ")";
  }
  void VisitExpr_(const CallNode *call) final {
    std::stringstream ss;
    ss << "Call_name(" << call->op << ")";
    m_dbgStr += ss.str();
  }
  void VisitExpr_(const ConstantNode *con) {
    m_dbgStr += "Const_name()";
  }

  private:
  std::string m_dbgStr;
};

struct BufferInfo {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  size_t max_bytes{0};
  /*! \brief The corresponding tensor type node. */
  const TensorTypeNode* ttype{nullptr};
  /*! \brief The storage id */
  int64_t storage_id{-1};

  /*! \brief virtual device index that corresponds to the device_type in
   * DLDevice. */
  int device_type{0};
  /*! \brief Number of bytes */
  size_t size{0};
  /*! \brief First use */
  int firstUse{-1};
  /*! \brief Last use */
  int lastUse{-1};
};

class LifetimeAnalyzer : public ExprVisitor {
 public:
  Map<Expr, Array<IntegerArray> > Plan(const Function& func) {
    prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);
    this->Run(func);

    // The value of smap contains two integer arrays where the first array
    // contains the planned storage ids and the second holds the device types.
    Map<Expr, Array<IntegerArray> > smap;
    int num_annotated_nodes = 0;
    int num_nodes = 0;

    for (const auto& kv : exprToBuffers_) {
      std::vector<Integer> storage_ids;
      std::vector<Integer> device_types;
      for (BufferInfo* tok : kv.second) {
        if (tok->device_type) {
          num_annotated_nodes++;
        }
        num_nodes++;
        storage_ids.push_back(tok->storage_id);
        device_types.push_back(tok->device_type);
      }
      smap.Set(GetRef<Expr>(kv.first), Array<IntegerArray>({storage_ids, device_types}));
    }
    // Either all or none of the nodes should be annotated.
    if (num_annotated_nodes != 0 && num_annotated_nodes != num_nodes) {
      LOG(FATAL) << num_annotated_nodes << " out of " << num_nodes
                 << "expressions are assigned with virtual device types. Either all "
                    "or none of the expressions are expected to be annotated.";
    }
    return smap;
  }

  /*! \return The internal token map */
  std::unordered_map<const ExprNode*, std::vector<BufferInfo*> > GetInitTokenMap(
      const Function& func) {
    node_device_map_ = CollectDeviceInfo(func);
    this->Run(func);
    return std::move(exprToBuffers_);
  }

  // run the visitor on a function.
  void Run(const Function& func) {
    printf("----- Visiting function\n");
    for (Var param : func->params) {
      CreateBuffer(param.operator->(), false);
    }
    // must always keep output alive.
    for (BufferInfo* tok : GetBuffers(func->body)) {
      tok->ref_counter += 1;
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    this->CreateBuffer(op, false);
  }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const TupleNode* op) final {
    std::vector<BufferInfo*> fields;
    for (Expr field : op->fields) {
      auto bufs = GetBuffers(field);
      fields.insert(fields.end(), bufs.begin(), bufs.end());
    }
    exprToBuffers_[op] = fields;
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    const auto& bufs = GetBuffers(op->tuple);
    ICHECK_LT(static_cast<size_t>(op->index), bufs.size());
    exprToBuffers_[op] = {bufs[op->index]};
  }

  void VisitExpr_(const IfNode* op) final {
    LOG(FATAL) << "if is not supported.";
  }

  void VisitExpr_(const LetNode* op) final {
    auto bufs = GetBuffers(op->value);
    exprToBuffers_[op->var.operator->()] = bufs;
    exprToBuffers_[op] = GetBuffers(op->body);
  }

  void VisitExpr_(const CallNode* op) final {
    std::string name = DbgVisitor().GetSimpleExprName(op);
    printf("----- Visiting call [[%s]] %i\n", name.c_str(), 0);

    // std::vector<StorageToken*> args;
    // create token for the call node.
    CreateBuffer(op, true);
    // for each input, visit argument token.
    for (Expr arg : op->args) {
      for (BufferInfo* tok : GetBuffers(arg)) {
        tok->ref_counter += 1;
      }
    }
    /*
    // check if there is orphaned output that can be released immediately.
    for (StorageToken* tok : exprToBuffers_.at(op)) {
      CheckForRelease(tok);
    }
    for (StorageToken* tok : args) {
      tok->ref_counter -= 1;
      CheckForRelease(tok);
    }
    */
  }

 private:
  /*!
   * \brief Get the necessary token.
   * \param expr The expression.
   * \return The corresponding token.
   */
  const std::vector<BufferInfo*>& GetBuffers(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = exprToBuffers_.find(expr.operator->());
    ICHECK(it != exprToBuffers_.end());
    return it->second;
  }
  /*!
   * \brief Populate the token map to set op's tokens
   * \param op The node to be processed.
   * \param can_realloc Whether we can re-allocate the memory.
   */
  void CreateBuffer(const ExprNode* op, bool can_realloc) {
    ICHECK(!exprToBuffers_.count(op));
    std::vector<BufferInfo*> tokens;
    int device_type =
        node_device_map_.count(GetRef<Expr>(op)) ? node_device_map_[GetRef<Expr>(op)]->value : 0;
    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        const auto* ttype = t.as<TensorTypeNode>();
        ICHECK(ttype);
        BufferInfo* token = arena_.make<BufferInfo>();
        token->ttype = ttype;
        token->device_type = device_type;
        tokens.push_back(token);
      }
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      ICHECK(ttype);
      BufferInfo* token = arena_.make<BufferInfo>();
      token->ttype = ttype;
      token->device_type = device_type;
      tokens.push_back(token);
    }
    exprToBuffers_[op] = tokens;
  }

 private:
  // allocator
  support::Arena arena_;
  /*! \brief internal prototype token map */
  std::unordered_map<const ExprNode*, std::vector<StorageToken*> > prototype_;
  /*! \brief internal token map */
  std::unordered_map<const ExprNode*, std::vector<BufferInfo*> > exprToBuffers_;
  Map<Expr, Integer> node_device_map_;
};

class FlattenVisitor : public ExprVisitor {
 public:
  std::unordered_map<const ExprNode*, int> GetExprTimes(const Function& func) {
    printf("----- Visiting function\n");
    VisitExpr(func->body);
    for (auto &it : exprToTime_) {
      it.second = opCount_ - it.second - 1;
    }
    return exprToTime_;
  }

  void VisitExpr_(const ConstantNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const CallNode* op) final {
    std::string name = DbgVisitor().GetSimpleExprName(op);
    printf("----- Visiting call [[%s]] %i\n", name.c_str(), opCount_);

    exprToTime_[op] = opCount_;
    opCount_++;

    // for each input, visit argument token.
    for (Expr arg : op->args) {
      VisitExpr(arg);
    }
  }

 private:
  int opCount_ = 0;
  std::unordered_map<const ExprNode*, int> exprToTime_;
};

/*!
 * \brief ceil(size/word_size) to get number of words.
 * \param size The original size.
 * \param word_size The element size.
 */
static size_t DivRoundUp(size_t size, size_t word_size) {
  return (size + word_size - 1) / word_size;
}

/*!
 * \brief Get the memory requirement.
 * \param prototype The prototype token.
 * \return The required memory size.
 */
static size_t GetMemorySize(const TensorTypeNode* ttype) {
  ICHECK(ttype != nullptr);
  size_t size = 1;
  for (IndexExpr dim : ttype->shape) {
    const int64_t* pval = tir::as_const_int(dim);
    ICHECK(pval != nullptr) << "Cannot allocate memory symbolic tensor shape " << ttype->shape;
    ICHECK_GE(*pval, 0) << "Cannot allocate memory for tensor with negative shape" << *pval;
    size *= static_cast<size_t>(pval[0]);
  }
  size *= DivRoundUp(ttype->dtype.bits() * ttype->dtype.lanes(), 8);
  return size;
}

class CallAnalyzer : public ExprVisitor {
 public:
  std::unordered_map<const ExprNode*, std::vector<BufferInfo*> > Run(const Function &func) {
    dg_ = DependencyGraph::Create(&arena_, func);
    node_device_map_ = CollectDeviceInfo(func);

    int labelId = 0;
    for (auto &node : dg_.post_dfs_order) {
      auto it = std::find_if(dg_.expr_node.begin(), dg_.expr_node.end(), [&](const std::pair<Expr, DependencyGraph::Node*> &val){
        return val.second == node;
      });
      if (it != dg_.expr_node.end()) {
        if (it->first.as<CallNode>()) {
          exprToLabel_[it->first] = "C" + std::to_string(labelId++);
        } else if (it->first.as<VarNode>()) {
          exprToLabel_[it->first] = "V" + std::to_string(labelId++);
        }
      }
    }

    for (Var param : func->params) {
      UpdateOrCreateBufs(param.get(), false, param.get());
    }

    VisitExpr(func->body);

    for (auto &exprToBufs : exprToBuffers_) {
      int firstUse = -2;
      int lastUse = -2;
      for (auto &buf : exprToBufs.second) {
        if (firstUse == -2) {
          firstUse = buf->firstUse;
          lastUse = buf->lastUse;
        } else {
          ICHECK(firstUse == buf->firstUse);
          ICHECK(lastUse == buf->lastUse);
        }
      }
      //std::string name = DbgVisitor().GetSimpleExprName(exprToBufs.first);
      //printf("----- Visiting call [[%s]] %i - %i\n", name.c_str(), firstUse, lastUse);
      printf("----- Visiting call [[%s]] %i - %i\n", GetLabel(exprToBufs.first).c_str(), firstUse, lastUse);
    }

    return exprToBuffers_;
  }

  std::string GetLabel(const Expr &expr) {
    return exprToLabel_[expr];
  }
  std::string GetLabel(const ExprNode *expr) {
    return exprToLabel_[GetRef<Expr>(expr)];
  }

  void VisitExpr_(const ConstantNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const VarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const FunctionNode* op) final {
    // do not recurse into sub function.
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const OpNode* op) final {
    // Do nothing.
  }

  void VisitExpr_(const CallNode* op) final {
    // Buffer used as output.
    UpdateOrCreateBufs(op, false, op);

    printf("Args of %s:\n", GetLabel(op).c_str());
    for (Expr arg : op->args) {
      printf(" %s\n", GetLabel(arg).c_str());
    }

    // Buffers used as input.
    for (Expr arg : op->args) {
      VisitExpr(arg);
      UpdateOrCreateBufs(arg.get(), true, op);
    }
  }

  void UpdateOrCreateBufs(const ExprNode* op, bool inputToNode, const ExprNode *useNode) {
    auto it = exprToBuffers_.find(op);
    if (it == exprToBuffers_.end()) {
      CreateBuffers(op);
      it = exprToBuffers_.find(op);
    }

    Expr expr = GetRef<Expr>(useNode);
    auto dgNode = dg_.expr_node[expr];
    auto itT = std::find(dg_.post_dfs_order.begin(), dg_.post_dfs_order.end(), dgNode);
    ICHECK(itT != dg_.post_dfs_order.end());
    auto t = static_cast<int>(std::distance(dg_.post_dfs_order.begin(), itT));

    for (auto &buf : it->second) {
      if (inputToNode) {
        buf->lastUse = std::max(buf->lastUse, t);
        printf("setting lastuse %i\n", buf->lastUse);
      } else {
        if (buf->firstUse == -1) {
          buf->firstUse = t;
        } else {
          buf->firstUse = std::min(buf->firstUse, t);
        }
        printf("setting firstuse %i\n", buf->firstUse);
      }
    }
  }

  void CreateBuffers(const ExprNode* op) {
    ICHECK(!exprToBuffers_.count(op));
    std::vector<BufferInfo*> bufs;
    Expr expr = GetRef<Expr>(op);
    int device_type =
        node_device_map_.count(expr) ? node_device_map_[expr]->value : 0;

    auto makeBuf = [&](Type t){
      const auto* ttype = t.as<TensorTypeNode>();
      ICHECK(ttype);
      BufferInfo *buf = arena_.make<BufferInfo>();
      buf->device_type = device_type;
      buf->size = GetMemorySize(ttype);
      return buf;
    };

    if (const auto* tuple_type = op->checked_type().as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        bufs.push_back(makeBuf(t));
      }
    } else {
      bufs.push_back(makeBuf(op->checked_type()));
    }
    exprToBuffers_[op] = bufs;
  }

 private:
  support::Arena arena_;
  DependencyGraph dg_;
  Map<Expr, Integer> node_device_map_;
  std::unordered_map<const ExprNode*, std::vector<BufferInfo*> > exprToBuffers_;

  std::map<Expr, std::string> exprToLabel_;
};

void OptimizeBuffers(Array<Expr> nodes, IntegerArray sizes, void *data)
{

}

Map<Expr, Array<IntegerArray>> GraphGetBuffers(const Function& func) {
  return {};
}

Map<Expr, Array<IntegerArray> > GraphPlanMemory(const Function& func) {
  auto bufs = GraphGetBuffers(func);

  CallAnalyzer().Run(func);

  // what to pass: buffers with size and lifetimes/conflicts

  auto pf = runtime::Registry::Get("tvm.relay.plan_memory");
  if (!pf) {
    printf("no pf\n");
  } else {
    //std::string str = (*pf)(func);
    //printf("got from poy: %s\n", str.c_str());
    Map<Expr, Array<IntegerArray>> out = (*pf)(func);
    printf("got map: %i\n", (int)out.size());
    for (auto p : out) {
      printf("Mapping: %i %i %i %i\n", (int)p.second[0][0], (int)p.second[1][0], (int)p.second[2][0], (int)p.second[3][0]);
    }
    return out;
  }

  return StorageAllocator().Plan(func);
}

TVM_REGISTER_GLOBAL("relay.backend.GraphPlanMemory").set_body_typed(GraphPlanMemory);
//TVM_REGISTER_GLOBAL("relay.backend.GraphGetBuffers").set_body_types(GraphGetBuffers);

}  // namespace relay
}  // namespace tvm
