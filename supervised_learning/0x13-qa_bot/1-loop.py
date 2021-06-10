#!/usr/bin/env python3
""" create a qa loop wihtout ml at first """


if __name__ == "__main__":
    words = ["exit", "quit", "goodbye", "bye"]
    exited = 0
    while (exited == 0):
        print("Q: ", end="")
        inp = input()
        if inp.lower() in words:
            print("A: Goodbye")
            exit()
        print("A: ")
