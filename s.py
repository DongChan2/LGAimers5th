import sys 
graph = []
blank =[]
for i in range(9):
    graph.append(list(map(int,sys.stdin.readline().rstrip().split())))  # 9 x 9 스도쿠 보드 초기화
for i in range(9):
    for j in range(9):
        if graph[i][j] == 0:
            blank.append((i,j))  # 비어있는 칸의 좌표 blank 리스트에 추가
def checkRow(x,a):
    # 특정 행에 같은 값이 있는지 없는지 체크
    for i in range(9):
        if a == graph[x][i]:  
            return False
    return True
def checkCol(y,a):
    # 특정 열에 같은 번호가 있는지 없는지 체크
    for i in range(9):
        if a == graph[i][y]:
            return False
    return True
def checkRect(x,y,a):
    # 3 x 3 행렬 내부에 같은 값이 있는지 체크
    nx = x//3 * 3 # 가장 가까운 3의 배수로 매핑
    ny = y//3 * 3
    for i in range(3):
        for j in range(3):
            if a == graph[nx + i][ny + j]:
                return False
    return True
def dfs(idx):
    if idx == len(blank):
        for i in range(9):
            print(" ".join(map(str,graph[i])))
        exit(0)
    for i in range(1,10):
        x,y = blank[idx] # blank에서 비어있는 값들의 좌표중 하나를 뽑는다.
        if checkRow(x,i) and checkCol(y,i) and checkRect(x,y,i):  # 세 조건이 모두 만족할때만
            graph[x][y] = i # 해당 위치에 i값 할당
            dfs(idx + 1) # 바로 다음 체크
            graph[x][y] = 0 # 다시 원복 시켜서 다른 경우의 수 체크
            
dfs(0)