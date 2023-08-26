import numpy
from Reversi.reversi_utils import Cell
from template import Agent
from Reversi.reversi_model import ReversiGameRule
import random, time, heapq
from json.encoder import INFINITY

CORNERS = [(0,0),(0,7),(7,0),(7,7)]
CORNERS_P = [(1,1),(1,6),(6,1),(6,6)]
Vmap = numpy.array([[500,-25,10,5,5,10,-25,500],
                    [-25,-200,1,1,1,1,-200,-25],
                    [10,1,3,2,2,3,1,10],
                    [5,1,2,1,1,2,1,5],
                    [5,1,2,1,1,2,1,5],
                    [10,1,3,2,2,3,1,10],
                    [-25,-200,1,1,1,1,-200,-25],
                    [500,-25,10,5,5,10,-25,500]])
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

    def PerformActionScore(self, state, action, agent_id):
        next_state = self.game_rules.generateSuccessor(state, action, agent_id)
        next_score = self.game_rules.calScore(next_state, agent_id)
        return next_state, next_score

    def GameEnd(self, state):
        if self.game_rules.getLegalActions(state,0) == ["Pass"] \
             and self.game_rules.getLegalActions(state,1) == ["Pass"]:
             return True
        else: return False
    
    def CountAll(self, state):
        score_a = self.game_rules.calScore(state, self.id)
        score_b = self.game_rules.calScore(state, 1 -self.id)
        return score_a + score_b
    
    def WinLose(self, state):
        score_a = self.game_rules.calScore(state, self.id)
        score_b = self.game_rules.calScore(state, 1 -self.id)
        if(score_a > score_b):
            return True
        else: return False

    def SelectAction(self,actions,game_state):
        self.game_rules.agent_colors = game_state.agent_colors
        myPQ = PriorityQueue()
        start_q = (game_state, [])
        myPQ.push(start_q, - self.heuristic(game_state))
        start_time = time.time()
        move = random.choice(actions)
        visited = set()
        count = 0
        max_score = -INFINITY

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
        return 0.02 * h1 + (-0.02) * h2 + 1.2 * h3 + (-0.9) * h4 + (-1.5) * h5 + 0.9 * h6 + (-1 * h7) + 0.6 * h8 + 0.25 * h9 + (-0.4 * h10)  + 1.1 * h11

    # def mapweightsum(self, board, mycolor):
    #     d = 0
    #     my_pieces = 0
    #     for i in range(8):
    #         for j in range(8):
    #             if board[i][j] == mycolor:
    #                 d += Vmap[i][j]
    #                 my_pieces += 1
    #     return my_pieces


    # def getmoves(self, state, c_id):
    #     h1 = len(self.game_rules.getLegalActions(state, c_id))
    #     h2 = len(self.game_rules.getLegalActions(state, 1 - c_id))
    #     return h1 - h2

    # def getstable(self, board, color, oppo_color):
    #     stable = [0,0,0]
    #     cor1 = [0,0,7,7]
    #     cor2 = [0,7,7,0]
    #     dir1 = [0,1,0,-1]
    #     dir2 = [1,0,-1,0]
    #     stop = [0,0,0,0]

    #     for i in range(4):
    #         if board[cor1[i]][cor2[i]] == color:
    #             stop[i] = 1
    #             stable[0] += 1
    #             for j in range(1,7):
    #                 if board[cor1[i]+dir1[i]*j][cor2[i]+dir2[i]*j] != color:
    #                     break
    #                 else:
    #                     stop[i] = j + 1
    #                     stable[1] += 1
    #     for i in range(4):
    #         if board[cor1[i]][cor2[i]] == color:
    #             for j in range(1,7-stop[i-1]):
    #                 if board[cor1[i]-dir1[i-1]*j][cor2[i]-dir2[i-1]*j] != color:
    #                     break
    #                 else:
    #                     stable[1] += 2
    #     age_pieces = 0
    #     oppo_pieces = 0
    #     if board[0][0] == Cell.EMPTY:
    #         if board[0][1] == color:
    #             age_pieces += 1
    #         elif board[0][1] == oppo_color:
    #             oppo_pieces += 1
    #         if board[1][1] == color:
    #             age_pieces += 1
    #         elif board[1][1] == oppo_color:
    #             oppo_pieces += 1
    #         if board[1][0] == color:
    #             age_pieces += 1
    #         elif board[1][0] == oppo_color:
    #             oppo_pieces += 1

    #     if board[0][7] == Cell.EMPTY:
    #         if board[0][6] == color:
    #             age_pieces += 1
    #         elif board[0][6] == oppo_color:
    #             oppo_pieces += 1
    #         if board[1][6] == color:
    #             age_pieces += 1
    #         elif board[1][6] == oppo_color:
    #             oppo_pieces += 1
    #         if board[1][7] == color:
    #             age_pieces += 1
    #         elif board[1][7] == oppo_color:
    #             oppo_pieces += 1

    #     if board[7][0] == Cell.EMPTY:
    #         if board[7][1] == color:
    #             age_pieces += 1
    #         elif board[7][1] == oppo_color:
    #             oppo_pieces += 1
    #         if board[6][1] == color:
    #             age_pieces += 1
    #         elif board[6][1] == oppo_color:
    #             oppo_pieces += 1
    #         if board[6][0] == color:
    #             age_pieces += 1
    #         elif board[6][0] == oppo_color:
    #             oppo_pieces += 1

    #     if board[7][7] == Cell.EMPTY:
    #         if board[6][7] == color:
    #             age_pieces += 1
    #         elif board[6][7] == oppo_color:
    #             oppo_pieces += 1
    #         if board[6][6] == color:
    #             age_pieces += 1
    #         elif board[6][6] == oppo_color:
    #             oppo_pieces += 1
    #         if board[7][6] == color:
    #             age_pieces += 1
    #         elif board[7][6] == oppo_color:
    #             oppo_pieces += 1
    #     stable[2] = (-5) * age_pieces + 2 * oppo_pieces
    #     #print(stable)
    #     return stable

    # def heuristic(self, state, count, c_id):
    #     age_color = self.game_rules.agent_colors[c_id]
    #     oppo_color = self.game_rules.agent_colors[1 - c_id]
    #     board = state.board

    #     moves = self.getmoves(state, c_id)
    #     stable = self.getstable(board, age_color, oppo_color)
    #     if (count < 24):
    #         value = self.mapweightsum(board, age_color) + (-1) * self.mapweightsum(board, oppo_color) + 500*moves
    #     else:
    #         value = self.mapweightsum(board, age_color) + (-1) * self.mapweightsum(board, oppo_color) + 20*moves + 100*(stable[0] + stable[1]) + 50 * stable[2]
    #     return int(value)



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