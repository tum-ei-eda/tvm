# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Fuse Seqence of operators and optimize"""
from typing import Callable, Optional, Union, Tuple

import tvm
from tvm import te
from tvm import tir
from ..te import extern
from ..tir import decl_buffer, ir_builder




def Fuseop(
    op_name: str,
    data: tvm.te.Tensor,
    filter: tvm.te.Tensor,
    bias: tvm.te.Tensor,
    axis: Optional[int] = None,
    dtype: Optional[int] = None,
    pool_size: Optional[Union[int, Tuple[int, int]]] = None,
    pool_strides: Optional[Union[int, Tuple[int, int]]] = None,
    pool_padding: Optional[Union[int, Tuple[int, int],Tuple[int, int, int, int]]] = None,
    pool_layout: Optional[str] = None,
    bias_axis: Optional[int] = None,
    padding: Optional[Union[int, Tuple[int, int],Tuple[int, int, int, int]]] = None,
    groups: Optional[int] = None,
    channels: Optional[int] = None,
    kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
    data_layout: Optional[str] = None,
    kernel_layout: Optional[str] = None,
) -> tvm.te.Tensor:
    """Fuse a sequence of operators (conv2d -> ... -> maxpool2d) and optimize.

    See Fuseseq for an example of use.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    filter: tvm.te.Tensor,
            The filter for the convolution operator.

    bias: tvm.te.Tensor,
            The bias for the bias operator.

    axis : int, optional
        Axis along which the operation is computed. The default (None) is to compute
        the cumulative operation over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are computed.
        If dtype is not specified, it defaults to the dtype of data.

    pool_size: Optional
        size of the maxpool operation.

    pool_strides: Optional
        strides of the maxpool operation.

    pool_padding: Optional
        padding of the maxpool operation.

    pool_layout: Optional
        layout of the maxpool data output.

    bias_axis: Optional
        axis of the bias operation on which the bias should be applied.

    padding: Optional
        padding of the convolution operation.

    groups: Optional
        groups of the convolution operation.

    channels: Optional
        number of output channels of the convolution operation

    kernel_size: Optional
        size of the convolution kernel.

    data_layout: Optional
        layout of the output data.

    kernel_layout: Optional
        layout of the kernel of the convolution operation.
        Default: HWIO

    Returns
    -------
    result : tvm.te.Tensor
          The result's size is defined by the kernelsize and stride of the convolution and maxpool operation. 
    """

    #redifine input shape
    shape = data.shape


    if axis is None:
        axis = 2
        if axis < 0:
            axis = len(shape) + axis

    #define output datatype
    if dtype is None or dtype == "":
        dtype = data.dtype

    #calculate outputshape
    out_shape = (shape[0], (shape[1] - (kernel_size[0] - 1))//2, (shape[2] - (kernel_size[1] - 1))//2, channels)

    # generate ir code for rollingbuffer
    def gen_ir(data_buf,fltr_buf,bias_buf,out_buf):

        #TODO: what if another datalayout
        #TODO: what if another kernellayout
        #TODO: if operation should be more general use the following parameters

        conv_padding= padding
        conv_layout=data_layout
        conv_groups=groups
        conv_channels=channels
        conv_kernel_size=kernel_size
        conv_kernel_layout=kernel_layout
        conv_out_dtype=dtype

        maxpool_strides=pool_strides
        maxpool_size=pool_size
        maxpool_padding=pool_padding
        maxpool_layout=pool_layout


        #define maximum axis after the convolution
        reduced_axis=[shape[1]-(conv_kernel_size[0]-1),shape[2]-(conv_kernel_size[1]-1)]



        #Setup irbuilder and buffers
        ib = ir_builder.create()
        data_buf = ib.buffer_ptr(data_buf)
        fltr_buf = ib.buffer_ptr(fltr_buf)
        bias_buf = ib.buffer_ptr(bias_buf)
        out_buf = ib.buffer_ptr(out_buf)
        #define shape and size of intermediate buffer
        im_shape=(reduced_axis[0]+2)

        # Create intermediate Buffer for the rolling buffer
        im_buf = ib.allocate(dtype,im_shape,name='Rolling_Buffer')

        #define calculation
        with ib.for_range(0, out_shape[3]-1, name='out_channels', dtype="int32", kind="serial") as c_out:
            with ib.for_range(0,reduced_axis[0]-1,name='im_height',dtype='int32',kind='serial') as h:
                #when height is increased by maxpool stride flush buffer
                with ib.if_scope((h % maxpool_strides[0]) == 0):
                    fill_level = 0
                with ib.for_range(0,reduced_axis[1]-1,name='im_width',dtype="int32",kind="serial") as w:
                    #calculate buffer iterator
                    i = (w + h*reduced_axis[0])%im_shape
                    #clear bufferelement
                    im_buf[i] = 0.0

                    with ib.for_range(0, shape[3]-1, name='in_channels',dtype="int32",kind="serial") as c_in:
                        with ib.for_range(0,conv_kernel_size[0]-1,name='kernel_height',dtype="int32",kind="serial") as k_h:
                            with ib.for_range(0,conv_kernel_size[1]-1,name='kernel_width',dtype="int32",kind="serial") as k_w:
                                _h=h+k_h
                                _w=w+k_w
                                #Convolution
                                im_buf[i]=tir.Add(im_buf[i],tir.Mul(data_buf[0,_h,_w,c_in],fltr_buf[k_h,k_w,c_in,c_out]))
                    #Bias_add
                    im_buf[i]=tir.Add(im_buf[i],bias_buf[c_out])
                    #Relu
                    im_buf[i]=tir.Max(im_buf[i],0.0)

                    #increase fill_level
                    fill_level=fill_level+1

                    #if max fill level reached execute maxpool
                    with ib.if_scope(fill_level>=im_shape):
                        #maxpool
                        out_buf[0,(h-1)//maxpool_strides[0],(w-1)//maxpool_strides[1],c_out]= tir.Max(tir.Max(im_buf[(i-1)%im_shape],im_buf[i%im_shape]),tir.Max(im_buf[(i+1)%im_shape],im_buf[(i+2)%im_shape]))
                        #decrease the fill level by 2
                        fill_level=fill_level-2


        return ib.get()

    # generate ir code for rollingbuffer with minimal size
    def gen_ir2(data_buf, fltr_buf, bias_buf, out_buf):

        #TODO: what if another datalayout
        #TODO: what if another kernellayout
        #TODO: if operation should be more general use the following parameters

        conv_padding = padding
        conv_layout = data_layout
        conv_groups = groups
        conv_channels = channels
        conv_kernel_size = kernel_size
        conv_kernel_layout = kernel_layout
        conv_out_dtype = dtype

        maxpool_strides = pool_strides
        maxpool_size = pool_size
        maxpool_padding = pool_padding
        maxpool_layout = pool_layout

        # define maximum axis after the convolution
        reduced_axis = [shape[1] - (conv_kernel_size[0] - 1), shape[2] - (conv_kernel_size[1] - 1)]

        # Setup irbuilder and buffers
        ib = ir_builder.create()
        data_buf = ib.buffer_ptr(data_buf)
        fltr_buf = ib.buffer_ptr(fltr_buf)
        bias_buf = ib.buffer_ptr(bias_buf)
        out_buf = ib.buffer_ptr(out_buf)
        # define shape and size of intermediate buffer
        im_shape = (4)

        # Create intermediate Buffer for the rolling buffer
        im_buf = ib.allocate(dtype, im_shape, name='Rolling_Buffer')

        # define calculation
        with ib.for_range(0, out_shape[3] - 1, name='out_channels', dtype="int32", kind="serial") as c_out:
            with ib.for_range(0, tir.indexdiv(reduced_axis[0],2) - 1, name='im_height', dtype='int32', kind='serial') as h_r:

                # when height is increased by maxpool stride flush buffer
                fill_level = 0

                with ib.for_range(0, tir.indexdiv(reduced_axis[1],2) - 1, name='im_width', dtype="int32", kind="serial") as w_r:

                    with ib.for_range(0,1, name='sub_im_height', dtype='int32', kind='serial') as y:
                        with ib.for_range(0, 1, name='sub_im_height', dtype='int32', kind='serial') as x:
                            h = h_r * 2+y
                            w = w_r * 2+x
                            # calculate buffer iterator
                            i = (x+y*2) % im_shape
                            # clear bufferelement
                            im_buf[i] = 0.0

                            with ib.for_range(0, shape[3] - 1, name='in_channels', dtype="int32", kind="serial") as c_in:
                                with ib.for_range(0, conv_kernel_size[0] - 1, name='kernel_height', dtype="int32",
                                                  kind="serial") as k_h:
                                    with ib.for_range(0, conv_kernel_size[1] - 1, name='kernel_width', dtype="int32",
                                                      kind="serial") as k_w:
                                        _h = h + k_h
                                        _w = w + k_w
                                        # Convolution
                                        im_buf[i] = tir.Add(im_buf[i],
                                                            tir.Mul(data_buf[0, _h, _w, c_in], fltr_buf[k_h, k_w, c_in, c_out]))
                            # Bias_add
                            im_buf[i] = tir.Add(im_buf[i], bias_buf[c_out])
                            # Relu
                            im_buf[i] = tir.Max(im_buf[i], 0.0)

                            # increase fill_level
                            fill_level = fill_level + 1

                            # if max fill level reached execute maxpool
                            with ib.if_scope(fill_level >= im_shape):
                                # maxpool
                                out_buf[0, (h - 1) // maxpool_strides[0], (w - 1) // maxpool_strides[1], c_out] = tir.Max(
                                    tir.Max(im_buf[(i - 1) % im_shape], im_buf[i % im_shape]),
                                    tir.Max(im_buf[(i + 1) % im_shape], im_buf[(i + 2) % im_shape]))
                                # decrease the fill level by 2
                                fill_level = 0

        return ib.get()




        return ib.get()

    out_buf = decl_buffer(out_shape,dtype,"out_buf")
    #if minimal memory footprint needed change gen_ir to gen_ir2
    return extern([out_shape],[data,filter,bias],lambda ins, outs: gen_ir(ins[0],ins[1],ins[2],outs[0]),dtype=dtype,out_buffers=[out_buf],name=op_name,tag=op_name)


def Fuseseq(
    data: tvm.te.Tensor,
    filter: tvm.te.Tensor,
    bias: tvm.te.Tensor,
    axis: Optional[int] = None,
    dtype: Optional[int] = None,
    pool_size: Optional[Union[int, Tuple[int, int]]] = None,
    pool_strides: Optional[Union[int, Tuple[int, int]]] = None,
    pool_padding: Optional[Union[int, Tuple[int, int],Tuple[int, int, int, int]]] = None,
    pool_layout: Optional[str] = None,
    bias_axis: Optional[int] = None,
    padding: Optional[Union[int, Tuple[int, int],Tuple[int, int, int, int]]] = None,
    groups: Optional[int] = None,
    channels: Optional[int] = None,
    kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
    data_layout: Optional[str] = None,
    kernel_layout: Optional[str] = None,
) -> tvm.te.Tensor:
    """Fuse a sequence of operators (conv2d -> ... -> maxpool2d) and optimize.

        Parameters
        ----------
        data : tvm.te.Tensor
            The input data to the operator.

        filter: tvm.te.Tensor,
                The filter for the convolution operator.

        bias: tvm.te.Tensor,
                The bias for the bias operator.

        axis : int, optional
            Axis along which the operation is computed. The default (None) is to compute
            the cumulative operation over the flattened array.

        dtype : string, optional
            Type of the returned array and of the accumulator in which the elements are computed.
            If dtype is not specified, it defaults to the dtype of data.

        pool_size: Optional
            size of the maxpool operation.

        pool_strides: Optional
            strides of the maxpool operation.

        pool_padding: Optional
            padding of the maxpool operation.

        pool_layout: Optional
            layout of the maxpool data output.

        bias_axis: Optional
            axis of the bias operation on which the bias should be applied.

        padding: Optional
            padding of the convolution operation.

        groups: Optional
            groups of the convolution operation.

        channels: Optional
            number of output channels of the convolution operation

        kernel_size: Optional
            size of the convolution kernel.

        data_layout: Optional
            layout of the output data.

        kernel_layout: Optional
            layout of the kernel of the convolution operation.
            Default: HWIO

        Returns
        -------
        result : tvm.te.Tensor
              The result's size is defined by the kernelsize and stride of the convolution and maxpool operation.
        """
    return Fuseop(
        data=data,
        filter=filter,
        bias=bias,
        op_name="Fusedops_Cifar10",
        axis=axis,
        dtype=dtype,
        pool_size = pool_size,
        pool_strides = pool_strides,
        pool_padding = pool_padding,
        pool_layout = pool_layout,
        bias_axis = bias_axis,
        padding = padding,
        groups = groups,
        channels = channels,
        kernel_size = kernel_size,
        data_layout = data_layout,
        kernel_layout = kernel_layout,

    )



