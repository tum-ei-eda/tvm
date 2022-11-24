from dis import Instruction
import re
import pandas as pd
import subprocess


def read_file():
    with open('/home/mohamedali/Documents/FP/projects/tvm/test/text_from_terminal.txt','r') as file:
        text = file.read()
    return text

def replace_extra_sting(text_file):
    text = re.sub(r'(?<=Node).*?(?=Total_time)', '',text_file, flags=re.DOTALL)
    text = text.replace("Node", "")
    text = re.sub(r'(?<=result).*?(?=]])', '',text, flags=re.DOTALL)
    text = text.replace("result]]", "")
    text = re.sub(r'(?<=Total_time).*?(?=\d)', ': ',text, flags=re.DOTALL) #filter wieghtspaces

    return text

def create_file(final_text):
    with open('/home/mohamedali/Documents/FP/projects/tvm/test/filtered_text', 'w') as file:
        file.write(final_text)
    
if __name__ == "__main__":
    output = subprocess.call(['/home/mohamedali/Documents/FP/projects/tvm/test/save_text.sh'])
    text_file = read_file()
    text_filtered = replace_extra_sting(text_file)
    print(text_filtered)
    create_file(text_filtered)
