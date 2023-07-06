def getEnvironment(environment):
    if environment == "env":
        from environment.env import Varikon
        return Varikon
    else:
        ValueError("Invalid Puzzle")