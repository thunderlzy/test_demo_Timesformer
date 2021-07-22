list = []
with open('name.txt', 'r') as f:
    for line in f:
        list.append(line.strip())
 
with open("name_done.txt", "w") as f:
    for item in sorted(list):
        f.writelines(item)
        f.writelines('\n')
    f.close()

