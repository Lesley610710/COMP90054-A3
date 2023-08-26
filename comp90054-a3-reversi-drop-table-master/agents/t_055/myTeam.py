import collections
import json
from sre_constants import IN
from Reversi.reversi_utils import Cell
from template import Agent
from Reversi.reversi_model import ReversiGameRule
import random, time, heapq
from json.encoder import INFINITY

Gamma = 0.9
Epsilon = 0.85
CORNERS = [(0,0),(0,7),(7,0),(7,7)]
CORNERS_P = [(1,1),(1,6),(6,1),(6,6)]

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rules = ReversiGameRule(2)
    
    def PerformAction(self, state, action, agent_id):
        next_state = self.game_rules.generateSuccessor(state, action, agent_id)
        return next_state

    def CalScore(self, state, agent_id):
        next_score = self.game_rules.calScore(state, agent_id)
        return next_score

    def CountAll(self, state):
        score_a = self.game_rules.calScore(state, self.id)
        score_b = self.game_rules.calScore(state, 1 -self.id)
        return score_a + score_b


    def PerformActionScore(self, state, action, agent_id):
        next_state = self.game_rules.generateSuccessor(state, action, agent_id)
        next_score = self.game_rules.calScore(state, agent_id)
        return next_state, next_score

    def GameEnd(self, state):
        if self.game_rules.getLegalActions(state,0) == ["Pass"] \
             and self.game_rules.getLegalActions(state,1) == ["Pass"]:
             return True
        else: return False


    def SelectAction(self,actions,game_state):
        self.game_rules.agent_colors = game_state.agent_colors
        self.start_time = time.time()
        move = random.choice(actions)

        countall = self.CountAll(game_state)
        
        if(countall >= 36):

            #remove corners and subcorners
            con_actions = list(set(actions).intersection(set(CORNERS)))
            if(len(con_actions) > 0):
                actions = con_actions
            con_actions = list(set(actions).difference(set(CORNERS_P)))
            if(len(con_actions) > 0):
                actions = con_actions
            sub_list = []
            cor1 = [0,0,7,7]
            cor2 = [0,7,7,0]
            dir1 = [0,1,0,-1]
            dir2 = [1,0,-1,0]
            for i in range(4):
                if game_state.board[cor1[i]][cor2[i]] == Cell.EMPTY:
                    for j in range(4):
                        sub_list.append((cor1[i] + dir1[j],cor2[i] + dir2[j]))
            con_actions = list(set(actions).difference(sub_list))
            if(len(con_actions) > 0):
                actions = con_actions

            move = self.SelectActionMcts(actions, game_state)

            return move
        else:

            return self.GreedyBFS(game_state, actions)
            

    def WinLose(self, state):
        score_a = self.game_rules.calScore(state, self.id)
        score_b = self.game_rules.calScore(state, 1 -self.id)
        if(score_a > score_b):
            return True
        else: return False


    def GreedyBFS(self,game_state, actions):
        myPQ = PriorityQueue()
        start_q = (game_state, [])
        myPQ.push(start_q, - self.heuristic(game_state))
        start_time = time.time()
        visited = set()
        count= 0
        max_score = -INFINITY
        move = random.choice(actions)
        changed = False
        # print(move)
        # print("-------------")

        #PQ is not empty
        while (not myPQ.isEmpty()) and (time.time() - start_time < 0.93):
            count += 1
            state, action_list = myPQ.pop()
            next_actions = self.game_rules.getLegalActions(state, self.id)

            if (not state in visited and time.time() - start_time < 0.93):
                visited.add(state)
                #end of game or time greater than 0.96
                if (self.GameEnd(state) and self.WinLose(state)) or (time.time() - start_time > 0.93):
                    curr_score = self.CalScore(state, self.id) + 0.5*self.heuristic(state)
                    if (curr_score > max_score):
                        max_score = curr_score
                        move = action_list[0]
                        changed = True
                        # print(count)
                        # print(self.heuristic(state))
                        # print(move, "\n")
                if (count == 1):
                    con_actions = list(set(next_actions).intersection(set(CORNERS)))
                    if(len(con_actions) > 0):
                        next_actions = con_actions
                    con_actions = list(set(next_actions).difference(set(CORNERS_P)))
                    if(len(con_actions) > 0):
                        next_actions = con_actions
                    sub_list = []
                    cor1 = [0,0,7,7]
                    cor2 = [0,7,7,0]
                    dir1 = [0,1,0,-1]
                    dir2 = [1,0,-1,0]
                    for i in range(4):
                        if game_state.board[cor1[i]][cor2[i]] == Cell.EMPTY:
                            for j in range(4):
                                sub_list.append((cor1[i] + dir1[j],cor2[i] + dir2[j]))
                    con_actions = list(set(actions).difference(sub_list))
                    if(len(con_actions) > 0):
                        next_actions = con_actions
                for action in next_actions:
                    if (time.time() - start_time) > 0.93:
                        break
                    next_list = action_list + [action]
                    next_state = self.PerformAction(state, action, self.id)
                    #rivals's turn
                    rival_next_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
                    rival_max_score = 0
                    for rival_action in rival_next_actions:
                        _, rival_next_score = self.PerformActionScore(next_state, rival_action, 1 - self.id)
                        if rival_next_score > rival_max_score:
                            rival_max_score = rival_next_score
                            rival_best_action = rival_action
                    next_state = self.PerformAction(next_state, rival_best_action, 1 - self.id)
                    
                    newN = (next_state, next_list)
                    myPQ.push(newN, -(self.heuristic(next_state)))
        if(changed == False):
            best_Q = -INFINITY
            con_actions = list(set(actions).intersection(set(CORNERS)))
            if(len(con_actions) > 0):
                actions = con_actions
            con_actions = list(set(actions).difference(set(CORNERS_P)))
            if(len(con_actions) > 0):
                actions = con_actions
            sub_list = []
            cor1 = [0,0,7,7]
            cor2 = [0,7,7,0]
            dir1 = [0,1,0,-1]
            dir2 = [1,0,-1,0]
            for i in range(4):
                if game_state.board[cor1[i]][cor2[i]] == Cell.EMPTY:
                    for j in range(4):
                        sub_list.append((cor1[i] + dir1[j],cor2[i] + dir2[j]))
            con_actions = list(set(actions).difference(sub_list))
            if(len(con_actions) > 0):
                actions = con_actions
            for action in actions:
                if time.time() - self.start_time > 0.975:
                    print("time out exit")
                    break
                curr_state = self.PerformAction(game_state, action, self.id)
                Q_val = self.heuristic(curr_state)
                if Q_val > best_Q:
                    best_Q = Q_val
                    move = action
        return move
            
    def heuristic(self, state):
        age_color = self.game_rules.agent_colors[self.id]
        oppo_color = self.game_rules.agent_colors[1 - self.id]
        features = []
        #next_state = self.PerformAction(state, action, self.id)
        #agent's pieces count
        agent_score = self.game_rules.calScore(state, self.id)
        h1 = agent_score/64
        features.append(h1)
        #rival's pieces count
        rival_score = self.game_rules.calScore(state, 1 - self.id)
        h2 = rival_score/64
        features.append(h2)
        #
        h3 = 0
        h4 = 0
        h7 = 0
        h8 = 0

        board = state.board
        #pieces around corner
        age_pieces = 0
        oppo_pieces = 0
        if board[0][0] == Cell.EMPTY:
            if board[0][1] == age_color:
                age_pieces += 1
            elif board[0][1] == oppo_color:
                oppo_pieces += 1
            if board[1][1] == age_color:
                age_pieces += 1
            elif board[1][1] == oppo_color:
                oppo_pieces += 1
            if board[1][0] == age_color:
                age_pieces += 1
            elif board[1][0] == oppo_color:
                oppo_pieces += 1

        if board[0][7] == Cell.EMPTY:
            if board[0][6] == age_color:
                age_pieces += 1
            elif board[0][6] == oppo_color:
                oppo_pieces += 1
            if board[1][6] == age_color:
                age_pieces += 1
            elif board[1][6] == oppo_color:
                oppo_pieces += 1
            if board[1][7] == age_color:
                age_pieces += 1
            elif board[1][7] == oppo_color:
                oppo_pieces += 1

        if board[7][0] == Cell.EMPTY:
            if board[7][1] == age_color:
                age_pieces += 1
            elif board[7][1] == oppo_color:
                oppo_pieces += 1
            if board[6][1] == age_color:
                age_pieces += 1
            elif board[6][1] == oppo_color:
                oppo_pieces += 1
            if board[6][0] == age_color:
                age_pieces += 1
            elif board[6][0] == oppo_color:
                oppo_pieces += 1

        if board[7][7] == Cell.EMPTY:
            if board[6][7] == age_color:
                age_pieces += 1
            elif board[6][7] == oppo_color:
                oppo_pieces += 1
            if board[6][6] == age_color:
                age_pieces += 1
            elif board[6][6] == oppo_color:
                oppo_pieces += 1
            if board[7][6] == age_color:
                age_pieces += 1
            elif board[7][6] == oppo_color:
                oppo_pieces += 1

        h5 = age_pieces
        h6 = oppo_pieces

        h11 = 0
        cor1 = [0,0,7,7]
        cor2 = [0,7,7,0]
        dir1 = [0,1,0,-1]
        dir2 = [1,0,-1,0]
        stop = [0,0,0,0]
        for i in range(4):
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == age_color:
                h3 += 1
                stop[i] = 1
                for j in range(1,7):
                    if board[cor1[i]+dir1[i]*j][cor2[i]+dir2[i]*j] != age_color:
                        break
                    else:
                        stop[i] = j + 1
                        h11 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == Cell.EMPTY and  state.board[CORNERS_P[i][0]][CORNERS_P[i][1]] == age_color:
                h4 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == oppo_color:
                h7 += 1
            if state.board[CORNERS[i][0]][CORNERS[i][1]] == Cell.EMPTY and  state.board[CORNERS_P[i][0]][CORNERS_P[i][1]] == oppo_color:
                h8 += 1
        for i in range(4):
            if board[cor1[i]][cor2[i]] == age_color:
                for j in range(1,7-stop[i-1]):
                    if board[cor1[i]-dir1[i-1]*j][cor2[i]-dir2[i-1]*j] != age_color:
                        break
                    else:
                        h11 += 1
        features.append(h3/4)
        features.append(h4/4)
        #Edge
        features.append(h5/4)
        features.append(h6/4)
        #Rival Corner
        features.append(h7/4)
        features.append(h8/4)
        #mobility
        h9 = len(self.game_rules.getLegalActions(state, self.id))
        h10 = len(self.game_rules.getLegalActions(state, 1 - self.id))
        features.append(h9/4)
        features.append(h10/4)
        #print(features)
        return 0.02 * h1 + (-0.02) * h2 + 1.2 * h3 + (-0.9) * h4 + (-0.8) * h5 + 0.9 * h6 + (-1 * h7) + 0.65 * h8 + 0.25 * h9 + (-0.4 * h10)  + 0.5 * h11


    def SelectActionMcts(self,actions,game_state):
        self.game_rules.agent_colors = game_state.agent_colors
        #count = 0

        solution = random.choice(actions)

        val_ro_dict = collections.defaultdict(dict)
        value_dict = dict()
        ncount_dict = dict()
        expanded_action_s = dict()
        parent_dict = dict()

        t_root_state = 'root'
        #whether the state is fully expanded
        def is_fully_expanded(t_state, actions):
            if t_state in expanded_action_s:
                expanded_actions =  expanded_action_s[t_state]
                return list(set(actions).difference(set(expanded_actions)))
            else:
                return actions
        

        def SelectBestAct(state, actions):
            stateLength = len(state)
            #print(val_ro_dict)
            maxp = max(val_ro_dict[state], key=lambda k: val_ro_dict[state][k])
            #print("maxp ", maxp)
            emptyTuple = tuple()
            if(maxp[stateLength] != 'P'):
                emptyTuple = emptyTuple + (int(maxp[stateLength]),) + (int(maxp[stateLength + 1]),)
                return emptyTuple
            else:
                return random.choice(actions)

            
        while time.time() - self.start_time < 0.96:
            #count += 1
            state = game_state
            new_actions = actions
            t_cur_state = t_root_state
            t_def_state = t_cur_state
            vis_queue = Queue()
            reward = 0

            #select
            #if fully expand, select a state to expand
            while len(is_fully_expanded(t_cur_state, new_actions)) == 0 and not self.GameEnd(state):
                if time.time() - self.start_time > 0.962:
                    # print(value_dict)
                    # print("MCT: ", count)
                    # print("val_ro_dict", val_ro_dict)
                    solution = SelectBestAct(t_root_state, actions)
                    #print(solution,"\n")
                    return solution
                if(random.uniform(0,1) < 1 - Epsilon) and (t_cur_state in val_ro_dict):
                    cur_action = SelectBestAct(t_cur_state, new_actions)
                else:
                    cur_action = random.choice(new_actions)
                next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
                q = (t_cur_state, cur_action)
                vis_queue.push(q)
                #rival's turn
                rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
                rival_max_score = 0
                rival_best_state = next_state
                for rival_action in rival_new_actions:
                    rival_next_state, rival_next_score = self.PerformActionScore(next_state, rival_action, 1 - self.id)
                    if rival_next_score > rival_max_score:
                        rival_max_score = rival_next_score
                        rival_best_state = rival_next_state
                        rival_best_action = rival_action
                #iteration
                t_def_state = t_cur_state
                t_cur_state = t_cur_state + str(cur_action[0]) + str(cur_action[1]) + str(rival_best_action[0]) + str(rival_best_action[1])
                new_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
                state = rival_best_state

            #Expand
            available_actions = is_fully_expanded(t_cur_state, new_actions)
            if(len(available_actions)) == 0:
                cur_action = random.choice(new_actions)
            else:
                cur_action = random.choice(available_actions)
            if t_cur_state in expanded_action_s:
                expanded_action_s[t_cur_state].append(cur_action)
            else:
                expanded_action_s[t_cur_state] = [cur_action]
            q = (t_cur_state, cur_action)
            vis_queue.push(q)
            next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
            #rival's turn
            rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
            rival_max_score = 0
            rival_best_state = next_state
            for rival_action in rival_new_actions:
                rival_next_state, rival_next_score = self.PerformActionScore(next_state, rival_action, 1 - self.id)
                if rival_next_score > rival_max_score:
                    rival_max_score = rival_next_score
                    rival_best_state = rival_next_state
                    rival_best_action = rival_action
            t_cur_state = t_cur_state + str(cur_action[0]) + str(cur_action[1]) + str(rival_best_action[0]) + str(rival_best_action[1])
            sim_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
            state = rival_best_state

            #Simulation
            #initialize length of the simulation process
            sim_length = 0
            while not self.GameEnd(state):
                sim_length += 1
                if time.time() - self.start_time > 0.962:
                    # print(value_dict)
                    # print("val_ro_dict", val_ro_dict)
                    solution = SelectBestAct(t_root_state, actions)
                    #print(solution,"\n")
                    #print(optimal_action)
                    return solution
                cur_action = random.choice(sim_actions)
                next_state = self.game_rules.generateSuccessor(state, cur_action, self.id)
                #rival's turn
                rival_new_actions = self.game_rules.getLegalActions(next_state, 1 - self.id)
                rival_best_action = random.choice(rival_new_actions)
                rival_best_state = self.game_rules.generateSuccessor(next_state, rival_best_action, 1 - self.id)
                #iteration
                sim_actions = self.game_rules.getLegalActions(rival_best_state, self.id)
                state = rival_best_state

            self_score = self.CalScore(state, self.id)
            rival_score = self.CalScore(state, 1 - self.id)
            if self_score > rival_score:
                reward = 100 + self_score - rival_score
            else:
                reward = -(100 + rival_score - self_score)

            #Backpropagate
            cur_value = reward*(Gamma **sim_length)
            while not vis_queue.isEmpty() and time.time() - self.start_time < 0.96:
                t_state, cur_action = vis_queue.pop()
                if t_state in ncount_dict:
                    ncount_dict[t_state] += 1
                    #if cur_value > value_dict[t_state]:
                    update_value = value_dict[t_state] + (1/ncount_dict[t_state])*(cur_value - Gamma * value_dict[t_state])
                    value_dict[t_state] = update_value
                    if len(t_state) == (len(t_def_state) + 4):
                        val_ro_dict[t_def_state][t_state] = update_value
                    elif t_state in parent_dict:
                        val_ro_dict[parent_dict[t_state]][t_state] = update_value
                else:
                    ncount_dict[t_state] = 1
                    value_dict[t_state] = cur_value
                    if len(t_state) == (len(t_def_state) + 4):
                        val_ro_dict[t_def_state][t_state] = cur_value
                        parent_dict[t_state] = t_def_state
                cur_value *= Gamma

        # print(time.time() - start_time)
        # print(value_dict)
        # print("val_ro_dict", val_ro_dict)
        # print(ChooseBest())
        solution = SelectBestAct(t_root_state, actions)
        #print(solution,"\n")
        return solution



class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def getMinimumPriority(self):
        return self.heap[0][0]

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)