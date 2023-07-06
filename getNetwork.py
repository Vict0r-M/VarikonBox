def getNetwork(puzzle, networkType):
    if puzzle == "varikon":
        if networkType == "simple":
            from networks.netSimple import VarikonNet
            return VarikonNet
        elif networkType == "residual":
            from networks.netRes import VarikonNet
            return VarikonNet
        elif networkType == "paper":
            from networks.netPaper import VarikonNet
            return VarikonNet
        else:
            ValueError("Invalid Network Type")
    else:
        ValueError("Invalid Puzzle")
