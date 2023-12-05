import heapq

graph_map = {
    'A': {'B': 'r', 'D': 'd'},
    'B': {'A': 'l', 'C': 'r', 'E': 'd'},
    'C': {'B': 'l', 'F': 'd'},
    'D': {'A': 'u', 'E': 'r', 'G': 'd'},
    'E': {'B': 'u', 'D': 'l', 'F': 'r'},
    'F': {'C': 'u', 'E': 'l', 'I': 'd'},
    'G': {'D': 'u', 'H': 'r'},
    'H': {'G': 'l', 'I': 'r'},
    'I': {'F': 'u', 'H': 'l'}
}



def amount_direction_of_turn(curr_d, next_d): #우회전 +, 좌회전 -
    if curr_d == next_d:
        return 0
        
    if (curr_d in ['u', 'd'] and next_d in ['u', 'd']) or (curr_d in ['l', 'r'] and next_d in ['l', 'r']):
        return 2
    else:
        if (curr_d == 'u' and next_d == 'r') or (curr_d == 'r' and next_d == 'd') or (curr_d == 'd' and next_d == 'l') or (curr_d == 'l' and next_d == 'u'):
            return 1
        elif (curr_d == 'u' and next_d == 'l') or (curr_d == 'l' and next_d == 'd') or (curr_d == 'd' and next_d == 'r') or (curr_d == 'r' and next_d == 'u'):
            return -1

def dijkstra(graph_param, destinations, init_direction):
    graph = graph_param
    result = [0, destinations[0]] # dist, path
    direction = init_direction
    for i in range(len(destinations)-1):
        distances = {node: float('inf') for node in graph}  # start로 부터의 거리 값을 저장하기 위함
        start = destinations[i]
        end = destinations[i+1]
        distances[start] = 0  # 시작 값은 0이어야 함
        queue = []
        heapq.heappush(queue, [distances[start], start])  # 시작 노드부터 탐색 시작 하기 위함.

        first = 1
        while queue:  # queue에 남아 있는 노드가 없으면 끝
            current_distance, path_until_now = heapq.heappop(queue)  # 탐색 할 노드, 거리를 가져옴.
            current_destination = path_until_now[-1]
            if first == 1:
                curr_dir = direction
                first = 0
            else:
                curr_dir = graph[path_until_now[-2]][current_destination][0]
            #print("curr d : %d , path until now : %s" % (current_distance, path_until_now))
            if current_destination == end:
                result[0] += current_distance; result[1] = result[1] + path_until_now[1:]
                direction = curr_dir
                break

            if distances[current_destination] < current_distance:  # 기존에 있는 거리보다 길다면, 볼 필요도 없음
                continue
      
            for new_destination, new_distance_str in graph[current_destination].items():
                new_distance = len(new_distance_str)
                #print("cd : %s, nd : %s" % (curr_dir, new_distance_str[0]))
                distance = current_distance + new_distance + abs(amount_direction_of_turn(curr_dir, new_distance_str[0])) # 해당 노드를 거쳐 갈 때 거리
                if distance < distances[new_destination]:  # 알고 있는 거리 보다 작으면 갱신
                    distances[new_destination] = distance
                    heapq.heappush(queue, [distance, path_until_now + new_destination])  # 다음 인접 거리를 계산 하기 위해 큐에 삽입
        
        
            #print("current q : " + str(queue))

    return result

#노드로 지정된 곳 밖에서 움직일 때... -> 이건 걍 맵만 바꿔주면 ok, 모든 판을 노드로
#처음 시작 위치

def path_to_movement_plan(path, graph, init_dir):
    plan = []
    curr_dir = init_dir
    next_dir = ""
    for i in range(len(path)-1):
        next_move = graph[path[i]][path[i+1]]
        for char in next_move:
            next_dir = char
            if curr_dir == next_dir:
                plan.append('s')
            elif amount_direction_of_turn(curr_dir, next_dir) < 0:
                plan.append('l')
                plan.append('s')
            elif amount_direction_of_turn(curr_dir, next_dir) > 0:
                plan.append('r')
                plan.append('s')
        plan.append('s')
        curr_dir = next_dir
    
    return plan



if __name__=="__main__":
    first_dir = 'd'
    destinations = ['A', 'G', 'J', 'N']
    result = dijkstra(graph_map, destinations, first_dir)
    print(result)

    movement_plan = path_to_movement_plan(result[1], graph_map, first_dir)
    print(movement_plan)

