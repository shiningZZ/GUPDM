import os


def getfiles():
    filenames=os.listdir('/root/autodl-tmp/underwater/data/Raw/test/input')
    print(filenames)
    return filenames



if __name__ == '__main__':

    if not os.path.exists('input.txt'):
        os.mknod('input.txt')

    a = getfiles()
    # a.spilt('')
    l = len(a)
    with open("input.txt", "w") as f:
        for i in range(l):
            print(a[i])
            x = a[i]
            f.write(x)
            f.write('\n')
        f.close()

    print()