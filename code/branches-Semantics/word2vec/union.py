import os
input_path = "C:/Users/Mr、key/Desktop/demo/" #此处填好自己的路径，注意最后的"/"

whole_file = [os.path.join(input_path,file) for file in os.listdir(input_path)]
content = []
#对于每一个路径，将其打开之后，使用readlines获取全部内容
for w in whole_file:
    with open(w,'rb') as f:
        content = content+f.readlines()

#构造输出的路径，和输入路径在同一个文件夹下，如果该文件夹内没有这个文件会自动创建
output_path = os.path.join(input_path,'result.txt')
#将内容写入文件
with open(output_path,'wb') as f:
    f.writelines(content)
