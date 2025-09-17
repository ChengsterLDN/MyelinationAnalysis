import os

#dir = ["C:\\Users\\jonat\\Myelination\\dataset\\test\\0", "C:\\Users\\jonat\\Myelination\\dataset\\test\\1", "C:\\Users\\jonat\\Myelination\\dataset\\test\\2", "C:\\Users\\jonat\\Myelination\\dataset\\test\\3"]

dir = ["C:\\Users\\jonat\\Myelination\\dataset\\train\\0", "C:\\Users\\jonat\\Myelination\\dataset\\train\\1", "C:\\Users\\jonat\\Myelination\\dataset\\train\\2", "C:\\Users\\jonat\\Myelination\\dataset\\train\\3"]
for i in dir:
    os.chdir(i)
    print(os.getcwd())

    for count, f in enumerate(os.listdir()):
        f_name, f_ext = os.path.splitext(f)
        f_name = "aaa" + str(count)

        new_name = f'{f_name}{f_ext}'
        os.rename(f, new_name)