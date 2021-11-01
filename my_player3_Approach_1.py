class InputOutputHandler:
        
    def readFile(self, n = 5, path = 'input.txt'):
        print("Trying to read the file: ", path)
        with open(path, 'r') as file:
            data = file.read().split('\n')
            data = [list(map(int, list(x))) for x in data]
            myPiece = data[0][0]
            previousBoard = data[1:n+1]
            currentBoard = data[n+1: 2*n + 1]
            return myPiece, previousBoard, currentBoard
    
    def writeFile(self, result, path = 'output.txt'):
        print("Trying to write to the file: ", path)
        with open(path, 'w') as file:
            if result == 'PASS':
                file.write(result)
            else:
                res = ','.join(map(str, result[:2]))
                file.write(res)


from copy import deepcopy

class GOHost:
    
    def __init__(self, player, previousBoard, currentBoard, n = 5):
        self.boardSize = n
        # self.firstBoard = currentBoard
        self.currentBoard = currentBoard
        self.previousBoard = previousBoard
        self.myPlayer = player
        self.currentPlayer = player
        ## Operations to find orthogonally adjacent points
        self.neighboringPointsOperations = [(-1, 0), (1, 0), (0, -1), (0, 1)] #up, down, left, right
    
    def isPresentOnBoard(self, location):
        ## Check if the location of the piece is not out of the board
        if (0 <= location[0] < self.boardSize) and (0 <= location[1] < self.boardSize):
            return True 
        else:
            return False
        
    def getNeighbors(self, location):
        ## Get all neighbors on the 4 sides
        neighborsList = []
        for xOperation, yOperation in self.neighboringPointsOperations:
            neighbor = (location[0] + xOperation, location[1] + yOperation)
            if self.isPresentOnBoard(neighbor):
                neighborsList.append(neighbor)
        return neighborsList
        
    def getNeighboringAllies(self, location, board, player):
        ## Get Neighbors that are allies, i.e., of the same type (black -1 or white -2)
        neighboringAllies = []
        neighbors = self.getNeighbors(location)
        for neighbor in neighbors:
            if board[neighbor[0]][neighbor[1]] == player:
                neighboringAllies.append(neighbor)
        return neighboringAllies
    
    def getAllAlliesInCluster(self, location, board, player):
        ## DFS to find all allies in the cluster
        ## BFS and DFS both will work
        ## Using DFS because it occupies lesser space as compared to BFS
        
        ##  BFS
        # allies = set()
        # if self.isPresentOnBoard(location):
            # queue = [location]
            # visited = set()
            # while queue:
                # piece = queue.pop(0)
                # allies.add(piece)
                # neighboringAllies = self.getNeighboringAllies(location = piece, board = board, player = player)
                # for neighborAlly in neighboringAllies:
                    # if neighborAlly not in visited:
                        # visited.add(piece)
                        # queue.append(neighborAlly)
                        
        ## DFS
        allies = set()
        # if self.isPresentOnBoard(location):
        stack = [location]
        visited = set()
        while stack:
            piece = stack.pop()
            allies.add(piece)
            neighboringAllies = self.getNeighboringAllies(location = piece, board = board, player = player)
            for neighborAlly in neighboringAllies:
                if neighborAlly not in visited and neighborAlly not in allies:
                    stack.append(neighborAlly)
                    visited.add(piece)

        allAllies = list(allies)
        return allAllies
    
    ## We can directly use getLiberties but this will reduce the complexity
    ## in some ways by stopping immediately after finding a single liberty
    def hasLiberty(self, location, board, player):
        ## Check if the current piece has at least one liberty or free space in the orthogonally adjacent points
        # if self.isPresentOnBoard(location):
        allAllies = self.getAllAlliesInCluster(location = location, board = board, player = player)
        for ally in allAllies:
            neighbors = self.getNeighbors(ally)
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                    return True
        return False    
    def getLiberties(self, location, board, player):
        ## Return the liberties for the current piece, i.e., free spaces in the orthogonally adjacent points
        liberties = set()
        # if self.isPresentOnBoard(location):
        allAllies = self.getAllAlliesInCluster(location = location, board = board, player = player)
        for ally in allAllies:
            neighbors = self.getNeighbors(ally)
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                        liberties.add(neighbor)
        
        libertiesList = list(liberties)        
        return libertiesList
    
    def getDeadPieces(self, board, player):
        ## Return the captured pieces or pieces that are already dead
        deadPieces = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == player and not self.hasLiberty(location = (i, j), board=board, player=player):
                    deadPieces.append((i, j))
        return deadPieces   
    
    def removePieces(self, board, deadPieces):
        ## Remove Pieces and return the list
        # newBoard = deepcopy(board)
        newBoard = board
        for piece in deadPieces:
            newBoard[piece[0]][piece[1]] = 0
        return newBoard
        
    
    def checkMoveAndRemoveDead(self, location, board, player):
        # if self.isPresentOnBoard(location):
        
        ## Place Piece
        board[location[0]][location[1]] = player
        ## Remove Dead Pieces
        opponent = 3 - player
        deadPieces = self.getDeadPieces(board = board, player = opponent)
        newBoard = self.removePieces(board = board, deadPieces = deadPieces)
        ## Return Previous Board, New Board After Placement and Removing Dead 
        ## Pieces, and number of dead pieces removed
        return board, newBoard, len(deadPieces) 
        
    ## Not Needed, python allows checking equality of 2-D matrices by simply writing previous board == board
    # def ruleKO(previousBoard, board):
        # for i in range(self.boardSize):
            # for j in range(self.boardSize):
                # if board[i][j] != previousBoard[i][j]:
                    # return False
        # return True
    
    def onEdge(self, i, j):
        if i == 0 or i == self.boardSize-1 or j == 0 or j == self.boardSize-1:
            return True
        else:
            return False
    
    def getLibertyBasedMoves(self, board, player):
        ## To get all the moves possible in connected clusters and liberties
        opponent = 3 - player
        allLibertyMoves = set()
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == player:
                    liberties = self.getLiberties(location=(i, j), board=board, player=player)
                    if len(liberties) == 1:
                        allLibertyMoves.update(liberties)
                        if self.onEdge(i, j):
                            availableLocations = set()
                            location = liberties[0]
                            neighbors = self.getNeighbors((location[0], location[1]))
                            for neighbor in neighbors:
                                if board[neighbor[0]][neighbor[1]] == 0:
                                    availableLocations.add(neighbor)
                                    
#                             availableLocations.update(
#                                 [neighbor for neighbor in self.getNeighbors((liberties[0][0], liberties[0][1])) if newBoard[neighbor[0]][neighbor[1]] == 0]
#                                                      )

                            if availableLocations:
                                allLibertyMoves.update(availableLocations)
                            
                elif board[i][j] == opponent:
                    liberties = self.getLiberties(location = (i, j), board=board, player=opponent)
                    allLibertyMoves.update(liberties)
        
        return list(allLibertyMoves)
                    
    def getLegalMovesBest(self, previousBoard, board, player):
        legalMoves = []
        ## Try all liberty based moves first
        allLibertyMoves = self.getLibertyBasedMoves(board = board, player = player)
        for move in allLibertyMoves:
            tempBoard = deepcopy(board)
            _, newBoard, deadPieces = self.checkMoveAndRemoveDead(location=(move[0], move[1]), board=tempBoard, player=player) 
            ## Check KO Rule and if the chosen move will have any liberty left, i.e., not a suicide
            if self.hasLiberty(location = (move[0], move[1]), board = newBoard, player = player) and newBoard != board and newBoard != previousBoard:
                legalMoves.append((move[0], move[1], deadPieces))
          
        
        if len(legalMoves) != 0:
            ## Sorted in non-increasing order of number of deadPieces
            legalMovesSorted = sorted(legalMoves, key = lambda x:x[2], reverse = True)
            return legalMovesSorted
        
        ## If no liberty based moves are possible, then try to pick any other possible move
        elif len(legalMoves) == 0:
            for i in range(self.boardSize):
                for j in range(self.boardSize):
                    if board[i][j] == 0:
                        tempBoard = deepcopy(board)
                        _, newBoard, deadPieces = self.checkMoveAndRemoveDead(location=(i, j), board=tempBoard, player=player)
                        if self.hasLiberty(location=(i, j), board=newBoard, player=player) and newBoard != board and newBoard != previousBoard:
                            legalMoves.append((i, j, deadPieces))
            return legalMoves
    


class AgentMiniMax:
    def __init__(self, path='input.txt'):
        ## Read the input.txt
        myPlayer, previousBoard, currentBoard = InputOutputHandler().readFile(5, path)
        self.myPlayer = myPlayer
        self.previousBoard = previousBoard
        self.currentBoard = currentBoard
        ## Create the host
        self.goHost = GOHost(player = self.myPlayer, 
                             previousBoard = self.previousBoard, 
                             currentBoard = self.currentBoard,
                             n = 5)
        # self.boardSize = self.goHost.boardSize
        self.boardSize = 5
        # self.komi = self.boardSize / 2
        self.komi = 2.5
        ## No. of levels in the minimax tree to be evaluated
        self.level = 4
        ## To calculate the score for black and white 
        self.blackScore = 0
        self.whiteScore = 0

    def evaluationFunction(self, board, player, deadPiecesBlack, deadPiecesWhite):
        
        # opponent = 3 - player
        # playerScore, opponentScore = 0, 0
        # if opponent == 2:
            # opponentScore = self.komi
        # elif player == 2:
            # playerScore = self.komi
        # playerEndangeredLiberty, opponentEndangeredLiberty = 0, 0
        # for i in range(self.boardSize):
            # for j in range(self.boardSize):
                # if board[i][j] == player:
                    # playerScore += 1
                    # liberties = self.goHost.getLiberties((i, j), board, player)
                    # if len(liberties) <= 1:
                        # playerEndangeredLiberty += 1
                # elif board[i][j] == opponent:
                    # opponentScore += 1
                    # liberties = self.goHost.getLiberties((i, j), board, opponent)
                    # if len(liberties) <= 1:
                        # opponentEndangeredLiberty += 1     
        # evaluation = (playerScore - opponentScore) + (opponentEndangeredLiberty - playerEndangeredLiberty)
        # + (deadPiecesOpponent*10 - deadPiecesPlayer*16)
        # return evaluation
        
        blackScore, whiteScore = 0, self.komi
        blackEndangeredLiberty, whiteEndangeredLiberty = 0, 0
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == 1:
                    blackScore += 1
                    liberties = self.goHost.getLiberties(location=(i, j), board=board, player=1)
                    if len(liberties) <= 1:
                        blackEndangeredLiberty += 1
                elif board[i][j] == 2:
                    whiteScore += 1
                    liberties = self.goHost.getLiberties(location=(i, j), board=board, player=2)
                    if len(liberties) <= 1:
                        whiteEndangeredLiberty += 1
                        
        if player == 1:
            evaluation = (blackScore - whiteScore) + (whiteEndangeredLiberty - blackEndangeredLiberty)
            + (deadPiecesWhite*10 - deadPiecesBlack*16)
        else:
            evaluation = (whiteScore - blackScore) + (blackEndangeredLiberty - whiteEndangeredLiberty)
            + (deadPiecesBlack*10 - deadPiecesWhite*16)
        return evaluation
        
        
    def maxMode(self, board, previousBoard, player, level, alpha, beta, boardWithoutRemovingDeadPieces):
        # opponent = 3 - player
        
        ## Calculate Current Scores of the Players
        # deadPiecesBlack = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = 1)
        # deadPiecesWhite = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = 2)
        
        deadPiecesPlayer = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = player)
        
        if player == 1:
            self.blackScore += len(deadPiecesPlayer)
            # self.blackScore += len(deadPiecesBlack)
            # playerScore = self.blackScore
            # opponentScore = self.whiteScore
        elif player == 2:
            self.whiteScore += len(deadPiecesPlayer)
            # self.whiteScore += len(deadPiecesWhite)
            # playerScore = self.whiteScore
            # opponentScore = self.blackScore
        
        if level < 1:
            value = self.evaluationFunction(board=board, 
                                            player=player,
                                            deadPiecesBlack = self.blackScore, 
                                            deadPiecesWhite = self.whiteScore)
            
            deadPiecesPlayer = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = player)
            if player == 1:
                self.blackScore -= len(deadPiecesPlayer)
                # self.blackScore -= len(deadPiecesBlack)
            elif player == 2:
                self.whiteScore -= len(deadPiecesPlayer)
                # self.whiteScore -= len(deadPiecesWhite)
                
            alphaMax = value
            availableMoves = []
            
        else:
            alphaMax = float('-inf')
            availableMoves = []
            legalMovesBest = self.goHost.getLegalMovesBest(previousBoard = previousBoard, board = board, player = player)
            if len(legalMovesBest) == 25:
                return 100, [(2, 2)]
            
            for move in legalMovesBest:
                tempBoard = deepcopy(board)
                boardWithoutRemovingDeadPieces, newBoard, numberOfDeadPieces = self.goHost.checkMoveAndRemoveDead(
                    (move[0], move[1]), tempBoard, player)
                score, steps = self.minMode(board = newBoard, 
                                            previousBoard = board, 
                                            player = 3 - player,
                                            level = level - 1,
                                            alpha = alpha,
                                            beta = beta,
                                            boardWithoutRemovingDeadPieces = boardWithoutRemovingDeadPieces)
                if score > alphaMax:
                    alphaMax = score
                    availableMoves = [move] + steps
                
                ######################################################
                if alphaMax > beta:
                    break
                
                alpha = max(alpha, alphaMax)
                
        return alphaMax, availableMoves
    
    
    def minMode(self, board, previousBoard, player, level, alpha, beta, boardWithoutRemovingDeadPieces):
        # opponent = 3 - player
        
        ## Calculate Current Scores of the Players
        # deadPiecesBlack = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = 1)
        # deadPiecesWhite = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = 2)
        
        deadPiecesPlayer = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = player)
        
        if player == 1:
            self.blackScore += len(deadPiecesPlayer)
            # self.blackScore += len(deadPiecesBlack)
            # playerScore = self.blackScore
            # opponentScore = self.whiteScore
        elif player == 2:
            self.whiteScore += len(deadPiecesPlayer)
            # self.whiteScore += len(deadPiecesWhite)
            # playerScore = self.whiteScore
            # opponentScore = self.blackScore
        
        if level < 1:
            value = self.evaluationFunction(board=board, 
                                            player=player, 
                                            deadPiecesBlack=self.blackScore, 
                                            deadPiecesWhite=self.whiteScore)
            
            deadPiecesPlayer = self.goHost.getDeadPieces(board = boardWithoutRemovingDeadPieces, player = player)
            if player == 1:
                self.blackScore -= len(deadPiecesPlayer)
                # self.blackScore -= len(deadPiecesBlack)
            elif player == 2:
                self.whiteScore -= len(deadPiecesPlayer)
                # self.whiteScore -= len(deadPiecesWhite)
                
            betaMin = value
            availableMoves = []
            
        else:
            betaMin = float('inf')
            availableMoves = []
            legalMovesBest = self.goHost.getLegalMovesBest(previousBoard = previousBoard, board = board, player = player)
            
            for move in legalMovesBest:
                tempBoard = deepcopy(board)
                boardWithoutRemovingDeadPieces, newBoard, numberOfDeadPieces = self.goHost.checkMoveAndRemoveDead(
                    (move[0], move[1]), tempBoard, player)
                score, steps = self.maxMode(board = newBoard, 
                                            previousBoard = board, 
                                            player = 3 - player,
                                            level = level - 1,
                                            alpha = alpha,
                                            beta = beta,
                                            boardWithoutRemovingDeadPieces = boardWithoutRemovingDeadPieces)
                if score < betaMin:
                    betaMin = score
                    availableMoves = [move] + steps
                 
                ####################################################################################
                if betaMin < alpha:
                # if betaMin <= alpha:
                    break
                    
                ####################################################################################
                # beta = min(beta, betaMin) ## CONFIRM
                
                if betaMin < beta:
                    alpha = betaMin
        
        return betaMin, availableMoves
    
    def play(self):
        score, steps = self.maxMode(board = self.currentBoard, 
                                    previousBoard = self.previousBoard, 
                                    player = self.myPlayer,
                                    level = self.level,
                                    alpha = float('-inf'),
                                    beta = float('inf'),
                                    boardWithoutRemovingDeadPieces = self.currentBoard)
        result = ""
        if len(steps) > 0:
            result = steps[0]
        else:
            result = 'PASS'
        InputOutputHandler().writeFile(result = result, path = 'output.txt')
        


if __name__ == '__main__':
    agent = AgentMiniMax(path='input.txt').play()




