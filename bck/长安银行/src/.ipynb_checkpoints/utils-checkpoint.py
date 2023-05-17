def peek(df ,line = None):
    print(df.shape)
    if line is None:
        print(df.head())
    else:
        print(df.head(line))
