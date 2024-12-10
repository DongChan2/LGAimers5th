import sys 
input = sys.stdin.readline
N=int(input())
row = [0 for _ in range(N)]

def is_promising(x):
    for i in range(x):
        if row[x] == row[i] or abs(row[x]-row[i]) == abs(x-i):
            return False
    return True

ans=0
def backtrack(x):
    global ans
    if x == N:
        ans+=1
        return 
    
    for i in range(N):
        row[x] = i
        if is_promising(x):
            backtrack(x+1)
        
backtrack(0)
print(ans)