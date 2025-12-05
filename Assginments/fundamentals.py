def five():
    ans = 10 + 3 * 2 ** 2 
    # 10 + 3 * 4
    # 10 + 12
    # 22
    print(ans)

def ten():
    val = input("Enter a value: \t")
    parts = val.split(".")
    interger = parts[0]
    decimal = parts[1]

    print(interger, decimal)

ten()

