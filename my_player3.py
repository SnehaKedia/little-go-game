#!/usr/bin/env python
# coding: utf-8

# ### IMPORTS

# In[1]:


from random import choice
import random
import sys
import time
import math
from copy import deepcopy


# ### INPUT OUTPUT HANDLER

# In[2]:


## Parts of this class definition have been taken from host.py
class InputOutputHandler:

    def readFile(self, n = 5, path = 'input.txt'):
        with open(path, 'r') as file:
            data = file.read().split('\n')
            data = [list(map(int, list(x))) for x in data]
            myPiece = data[0][0]
            previousBoard = data[1:n+1]
            currentBoard = data[n+1: 2*n + 1]
            return myPiece, previousBoard, currentBoard
    
    def writeFile(self, result, path = 'output.txt'):
        with open(path, 'w') as file:
            if result == 'PASS':
                file.write(result)
            else:
                res = ','.join(map(str, result[:2]))
                file.write(res)


# ### HOST ENTITY

# In[3]:


## Parts of this class definition have been taken from host.py

class GOHost:
    
    def __init__(self, player, previousBoard, currentBoard, n = 5):

        self.boardSize = n
        # self.BlackMove = True
        # self.nMove = 0
        ## self.maxMove = n * (n - 1)
        # self.maxMove = 24
        # self.verbose = False
        ## self.komi = self.boardSize / 2
        self.komi = 2.5
        
        ## Define the Board State
        self.myPlayer = player
        self.previousBoard = previousBoard
        self.board = currentBoard
        self.deadPieces = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if previousBoard[i][j] == player and currentBoard[i][j] != player:
                    self.deadPieces.append((i, j))
        
        ## Operations to find orthogonally adjacent points
        # self.neighboringPointsOperations = [(-1, 0), (1, 0), (0, -1), (0, 1)] #up, down, left, right
        
    ############################################################
    
    # def isPresentOnBoard(self, location):
        # ## Check if the location of the piece is not out of the board
        # if (0 <= location[0] < self.boardSize) and (0 <= location[1] < self.boardSize):
            # return True 
        # else:
            # return False
    
    def compareBoard(self, board1, board2):
        ## Check if two boards are similar
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board1[i][j] != board2[i][j]:
                    return False
        return True
    
    def copyBoard(self):
        ## Creates a deepcopy of the board
        return deepcopy(self)
    
    def updateBoard(self, newBoard):
        ## Updates the existing board
        self.board = newBoard
        
    ############################################################
        
    def getNeighbors(self, location, board = None):
        ## Get all neighbors on the 4 sides
        if board == None:
            board = self.board
        neighborsList = []
        i = location[0]
        j = location[1]
        
        # for xOperation, yOperation in self.neighboringPointsOperations:
            # neighbor = (location[0] + xOperation, location[1] + yOperation)
            # if self.isPresentOnBoard(neighbor):
                # neighborsList.append(neighbor)
                
        if i > 0: neighborsList.append((i-1, j))
        if i < len(board) - 1: neighborsList.append((i+1, j))
        if j > 0: neighborsList.append((i, j-1))
        if j < len(board) - 1: neighborsList.append((i, j+1))
            
        return neighborsList
        
    def getNeighboringAllies(self, location, board = None):
        if board == None:
            board = self.board
        ## Get Neighbors that are allies, i.e., of the same type (black -1 or white -2)
        neighboringAllies = []
        neighbors = self.getNeighbors(location = location, board = board)
        for neighbor in neighbors:
            if board[neighbor[0]][neighbor[1]] == board[location[0]][location[1]]:
                neighboringAllies.append(neighbor)
        return neighboringAllies
    
    def getAllAlliesInCluster(self, location, board = None):
        # if board == None:
            # board = self.board
            
        ## DFS to find all allies in the cluster
        ## BFS and DFS both will work
        ## Using DFS because it occupies lesser space as compared to BFS
        
        ##  BFS
        # allies = set()
        # queue = [location]
        # while queue:
            # piece = queue.popleft(0)
            # allies.add(piece)
            # neighboringAllies = self.getNeighboringAllies(location = piece, board = board)
            # for neighborAlly in neighboringAllies:
                # if neighborAlly not in queue and neighborAlly not in allies:
                    # queue.append(neighborAlly)    
        ## DFS
        allies = set()
        stack = [location]
        while stack:
            piece = stack.pop()
            allies.add(piece)
            neighboringAllies = self.getNeighboringAllies(location = piece, board = board)
            for neighborAlly in neighboringAllies:
                if neighborAlly not in stack and neighborAlly not in allies:
                    stack.append(neighborAlly)

        allAllies = list(allies)
        return allAllies
    
    ## We can directly use getLiberties but this will reduce the complexity
    ## in some ways by stopping immediately after finding a single liberty
    
    def hasLiberty(self, location, board = None):
        if board == None:
            board = self.board
        ## Check if the current piece has at least one liberty or free space in the orthogonally adjacent points
        allAllies = self.getAllAlliesInCluster(location = location, board = board)
        for ally in allAllies:
            neighbors = self.getNeighbors(location = ally)
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                    return True
        return False
    
    ############################################################
    
    def getLiberties(self, location, board = None):
        if board == None:
            board = self.board
        ## Return the liberties for the current piece, i.e., free spaces in the orthogonally adjacent points
        liberties = set()
        # if self.isPresentOnBoard(location):
        allAllies = self.getAllAlliesInCluster(location = location, board = board)
        for ally in allAllies:
            neighbors = self.getNeighbors(location = ally)
            for neighbor in neighbors:
                if board[neighbor[0]][neighbor[1]] == 0:
                        liberties.add(neighbor)
        
        libertiesList = list(liberties)        
        return libertiesList
    
    ############################################################
    
    def getDeadPieces(self, player, board = None):
        if board == None:
            board = self.board
        ## Return the captured pieces or pieces that are already dead
        deadPieces = []
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == player and not self.hasLiberty(location = (i, j), board = board):
                    deadPieces.append((i, j))
        return deadPieces   
    
    def removeDeadPieces(self, player):
        ## Remove Dead Pieces and return the Dead Pieces List
        deadPieces = self.getDeadPieces(player = player)
        if not deadPieces:
            return []
        self.removeCertainPieces(deadPieces)
        return deadPieces
        
    def removeCertainPieces(self, pieces, board = None):
        board = self.board
        for piece in pieces:
            board[piece[0]][piece[1]] = 0
        self.updateBoard(board)
        # self.board = board
        
    ## Parts of this function have been taken from host.py
    def checkValidMove(self, location, player, board = None):
        if board == None:
            board = self.board
        
        ## Check if location is on board
        # if not self.isPresentOnBoard(location):
            # return False
            
        if not (location[0] >= 0 and location[0] < len(board)):
            return False
        if not (location[1] >= 0 and location[1] < len(board)):
            return False
        
        ## Check if location is not empty and has a piece already
        if board[location[0]][location[1]] != 0:
            return False
        
        ## Copy the board for checking purposes
        tempGo = self.copyBoard()
        tempBoard = tempGo.board
        
        ## No liberty left, it is a suicide move and hence not permitted
        tempBoard[location[0]][location[1]] = player
        tempGo.updateBoard(tempBoard)
        # tempGo.board = tempBoard 
        if tempGo.hasLiberty(location):
            return True
        
        ## If liberty not found, remove the dead pieces of opponent and try again
        tempGo.removeDeadPieces(3 - player)
        if not tempGo.hasLiberty(location):
            return False
        else:
            ## Check for KO rule (same board after repeating the placement)
            if self.deadPieces and self.compareBoard(self.previousBoard, tempGo.board):
                return False
        
        return True
    
    ## Parts of this function have been taken from host.py
    def score(self, player):
        ## Calculate the score for a player which is the number of pieces on the board
        board = self.board
        count = 0
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == player:
                    count += 1
        return count


# ### AI AGENT WITH MINIMAX ALGORITHM

# In[5]:


class AgentMiniMax:
    def __init__(self, myPlayer, previousBoard, currentBoard, boardSize, goHost):
        self.boardSize = boardSize
        self.myPlayer = myPlayer
        self.previousBoard = previousBoard
        self.board = currentBoard
        ## Create the player host object
        self.goHost = goHost
        ## No. of levels in the minimax tree to be evaluated
        self.level = 4
        self.opponentLevel = 1
        
        
    def getPieceLocations(self, board, player):
        ## Get all the locations of a particular piece/player on the board
        locations = []
        for i in range(self.goHost.boardSize):
            for j in range(self.goHost.boardSize):
                if board[i][j] == player:
                    locations.append((i,j))
        return locations
        
    def getWinningScores(self):
        ## Get the current score of the Black and White players
        scoreWhite = self.goHost.score(2)
        scoreBlack = self.goHost.score(1)
        return scoreWhite + self.goHost.komi, scoreBlack
    
    def evaluationFunction1(self, board, player):
        ## Difference between number of stones of the player and the opponent
        scoreWhite, scoreBlack = self.getWinningScores()
        if player == 1:
            score = scoreBlack - scoreWhite
        elif player == 2:
            score = scoreWhite - scoreBlack
        return score
    
    def evaluationFunction(self, board, player):
        blackScore, whiteScore = 0, self.goHost.komi
        blackEndangeredLiberty, whiteEndangeredLiberty = 0, 0
        deadPiecesBlack = len(self.goHost.getDeadPieces(player = 1, board = board))
        deadPiecesWhite = len(self.goHost.getDeadPieces(player = 2, board = board))
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i][j] == 1:
                    blackScore += 1
                    liberties = self.goHost.getLiberties(location=(i, j), board=board)
                    if len(liberties) <= 1:
                        blackEndangeredLiberty += 1
                elif board[i][j] == 2:
                    whiteScore += 1
                    liberties = self.goHost.getLiberties(location=(i, j), board=board)
                    if len(liberties) <= 1:
                        whiteEndangeredLiberty += 1
                        
        if player == 1:
            evaluation = (blackScore - whiteScore) + (whiteEndangeredLiberty - blackEndangeredLiberty)
            + (deadPiecesWhite*10 - deadPiecesBlack*16)
        else:
            evaluation = (whiteScore - blackScore) + (blackEndangeredLiberty - whiteEndangeredLiberty)
            + (deadPiecesBlack*10 - deadPiecesWhite*16)
        return evaluation
    
    def getLegalMoves(self, player, board = None):
        ## Get Possible/Legal Moves for Player
        legalMoves = []
        for i in range(self.goHost.boardSize):
            for j in range(self.goHost.boardSize):
                if self.goHost.checkValidMove(location = (i, j), player = player, board = board):
                    legalMoves.append((i, j))
        random.shuffle(legalMoves)
        return legalMoves
    
    def checkLegalMove(self, move, player):
        ## Check if a move is legal for the player
        if move in self.getLegalMoves(player = player, board = None):
            return True
        else:
            return False
        
    def playMove(self, move, player, board):
        ## If legal move, place the stone
        if self.checkLegalMove(move = move, player = player):
            self.goHost.previousBoard = deepcopy(board)
            board[move[0]][move[1]] = player
            self.goHost.board = board
        return board
    
    def getOpponentLegalMoves(self, oppGoHost, player):  ################# removed board = None
        ## Get Legal Moves for the Opponent
        legalMoves = []
        for i in range(oppGoHost.boardSize):
            for j in range(oppGoHost.boardSize):
                if oppGoHost.checkValidMove(location = (i, j), player = player, board = None):
                    legalMoves.append((i, j))
        random.shuffle(legalMoves)
        return legalMoves
        
    def checkOpponentLegalMove(self, oppGoHost, move, player):
        ## Check if a move is legal for the player
        if move in self.getOpponentLegalMoves(oppGoHost = oppGoHost, player = player):
            return True
        else:
            return False
    
    def playOpponentMove(self, oppGoHost, move, player, board):
        ## If legal move, place the stone
        if self.checkOpponentLegalMove(oppGoHost = oppGoHost, move = move, player = player):
            oppGoHost.previousBoard = deepcopy(board)
            board[move[0]][move[1]] = player
            oppGoHost.board = board
        return board
    
    ## Player Min Node Minimax
    def minNode(self, player, level, alpha, beta, start, board):
        newBoard = deepcopy(board)
        # betaMin = math.inf
        betaMin = float('inf')
        moves = self.getLegalMoves(player = player, board = None)
        end = time.time()
        
        if len(moves) == 0 or level == 0 or end-start > 8.5:
            return (-1, -1), self.evaluationFunction(board = newBoard, player = player)
        
        else:
            for move in moves:
                boardToPass = deepcopy(board)
                newBoard = self.playMove(move = move, player = player, board = boardToPass)
                self.goHost.removeDeadPieces(player = 3 - player)
                if player == 1:
                    nextPlayer = 2
                elif player == 2:
                    nextPlayer = 1
                newMove, newScore = self.maxNode(player =  nextPlayer, level = level - 1, alpha = alpha, 
                                                 beta = beta, start = start, board = newBoard)
                if newScore < betaMin:
                    betaMin = newScore
                    bestMove = move
                beta = min(newScore, beta)
                ## Alpha - Beta Pruning
                if beta <= alpha:
                    break
            return bestMove, betaMin
    
    ## Player Max Node Minimax
    def maxNode(self, player, level, alpha, beta, start, board):
        end = time.time()
        newBoard = deepcopy(board)
        # alphaMax = -math.inf
        alphaMax = float('-inf')
        moves = self.getLegalMoves(player = player, board = None)
        piecesToRemove = []
        for move in moves:
            self.goHost.board[move[0]][move[1]] = player
            opponentMoves = self.getLegalMoves(player = 3 - player, board = None) ######## board = self.goHost.board
            for oppMove in opponentMoves:
                self.goHost.board[oppMove[0]][oppMove[1]] = 3 - player
                deadPieces = self.goHost.getDeadPieces(player = player)
                self.goHost.board[oppMove[0]][oppMove[1]] = 0
                if (move in deadPieces) and (move not in piecesToRemove):
                    piecesToRemove.append(move)
            self.goHost.board[move[0]][move[1]] = 0     
        
        for piece in piecesToRemove:
            if piece in moves:
                moves.remove(piece)
                
        if len(moves) == 0 or level == 0 or end - start > 8.5:
            return (-1, -1), self.evaluationFunction(board = newBoard, player = player)
        
        else:
            for move in moves:
                boardToPass = deepcopy(board)
                newBoard = self.playMove(move = move, player = player, board = boardToPass)
                self.goHost.removeDeadPieces(player = 3 - player)
                if player == 1:
                    nextPlayer = 2
                elif player == 2:
                    nextPlayer = 1
                newMove, newScore = self.minNode(player =  nextPlayer,
                                                 level = level - 1, 
                                                 alpha = alpha, 
                                                 beta = beta, 
                                                 start = start, 
                                                 board = newBoard)
                if newScore > alphaMax:
                    alphaMax = newScore
                    bestMove = move
                alpha = max(newScore, alpha)
                ## Alpha - Beta Pruning
                if beta <= alpha:
                    break
            return bestMove, alphaMax
        
    ## Select Move Based on Minimax
    def selectMiniMaxMove(self, player, board):
        start = time.time()
        bestMove, score = self.maxNode(player =  player, 
                                       level = self.level, 
                                       alpha = float('-inf'), 
                                       beta = float('inf'), 
                                       start = start, 
                                       board = board)
        return bestMove, score
    
        
    ## Opponent Min Node Minimax
    def opponentMinNode(self, oppGoHost, player, level, alpha, beta, start, board):
        newBoard = deepcopy(board)
        # betaMin = math.inf
        betaMin = float('inf')
        moves = self.getOpponentLegalMoves(oppGoHost = oppGoHost, player = player)
        end = time.time()
        
        if len(moves) == 0 or level == 0 or end-start > 8.5:
            return (-1, -1), self.evaluationFunction(board = newBoard, player = player)
        
        else:
            for move in moves:
                boardToPass = deepcopy(board)
                newBoard = self.playOpponentMove(oppGoHost = oppGoHost, move = move, player = player, board = boardToPass)
                oppGoHost.removeDeadPieces(player = 3 - player)
                if player == 1:
                    nextPlayer = 2
                elif player == 2:
                    nextPlayer = 1
                newMove, newScore = self.opponentMaxNode(oppGoHost = oppGoHost,
                                                         player =  nextPlayer, 
                                                         level = level - 1, 
                                                         alpha = alpha, 
                                                         beta = beta, 
                                                         start = start,
                                                         board = newBoard)
                if newScore < betaMin:
                    betaMin = newScore
                    bestMove = move
                beta = min(newScore, beta)
                ## Alpha - Beta Pruning
                if beta <= alpha:
                    break
            return bestMove, betaMin
    
    ## Opponent Max Node Minimax
    def opponentMaxNode(self, oppGoHost, player, level, alpha, beta, start, board):
        end = time.time()
        newBoard = deepcopy(board)
        # alphaMax = -math.inf
        alphaMax = float('-inf')
        moves = self.getOpponentLegalMoves(oppGoHost = oppGoHost, player = player)
        
        if len(moves) == 0 or level == 0 or end - start > 8.5:
            return (-1, -1), self.evaluationFunction(board = newBoard, player = player)
        
        else:
            for move in moves:
                boardToPass = deepcopy(board)
                newBoard = self.playOpponentMove(oppGoHost = oppGoHost, 
                                                 move = move, player = player, board = boardToPass)
                oppGoHost.removeDeadPieces(player = 3 - player)
                if player == 1:
                    nextPlayer = 2
                elif player == 2:
                    nextPlayer = 1
                newMove, newScore = self.opponentMinNode(oppGoHost = oppGoHost, 
                                                         player =  nextPlayer, 
                                                         level = level - 1, 
                                                         alpha = alpha, 
                                                         beta = beta, 
                                                         start = start,
                                                         board = newBoard)
                if newScore > alphaMax:
                    alphaMax = newScore
                    bestMove = move
                alpha = max(newScore, alpha)
                ## Alpha - Beta Pruning
                if beta <= alpha:
                    break
            return bestMove, alphaMax
        
    ## Select Opponent Move Based on Minimax
    def selectOpponentMiniMaxMove(self, oppGoHost, player, board):
        start = time.time()
        bestMove, score = self.opponentMaxNode(oppGoHost = oppGoHost,
                                               player =  player, 
                                               level = self.opponentLevel, 
                                               alpha = float('-inf'), 
                                               beta = float('inf'), 
                                               start = start, 
                                               board = board)
        return bestMove, score
    
    ## Main Function to process the next move to take
    ## First, get the number of pieces we can conquer in if we play a move and based on that return the move
    ## If no such move then remove dead pieces from the board and check again, if still no move then pass
    ## Now decide what can be opponent's next move, if our piece is getting captured then we do not play that move
    ## We end up occupying opponent's liberties. Then, we choose and play the move according to our minimax algorithm.
    
    def getNextStep(self):
        
        player = self.myPlayer
        
        ## Count the number of opponents we can conquer for every empty space available on board
        ## In this way we cab select the best move (inspired from Killer Heuristic Method)
        
        freeSpaces = []
        for i in range(self.goHost.boardSize):
            for j in range(self.goHost.boardSize):
                if self.goHost.board[i][j] == 0:
                    freeSpaces.append((i,j)) 
        
        conquests = dict()
        for space in freeSpaces:
            self.goHost.board[space[0]][space[1]] = player
            deadPieces = self.goHost.getDeadPieces(player = 3 - player)
            self.goHost.board[space[0]][space[1]] = 0
            if len(deadPieces) >= 1:
                conquests[space] = len(deadPieces)
                
        sortedConquests = sorted(conquests, key = conquests.get, reverse = True)
        
        for conquest in sortedConquests:
            tempBoard = deepcopy(self.goHost.board)
            tempBoard[conquest[0]][conquest[1]] = player
            deadPieces = self.goHost.getDeadPieces(player = 3 - player, board = tempBoard)
            for deadp in deadPieces:
                tempBoard[deadp[0]][deadp[1]] = 0
            if conquest != None and self.goHost.previousBoard != tempBoard:
                return conquest
            
        ## Remove Dead Moves and see if it is a PASS
        
        moves = self.getLegalMoves(player = player, board = None)
        piecesToRemove = []
        for move in moves:
            self.goHost.board[move[0]][move[1]] = player
            opponentMoves = self.getLegalMoves(player = 3 - player, board = self.goHost.board)
            for opm in opponentMoves:
                self.goHost.board[opm[0]][opm[1]] = 3 - player
                deadPieces = self.goHost.getDeadPieces(player = player)
                self.goHost.board[opm[0]][opm[1]] = 0
                if move in deadPieces:
                    piecesToRemove.append(move)
            self.goHost.board[move[0]][move[1]] = 0
        
        for piece in piecesToRemove:
            if piece in moves:
                moves.remove(piece)
                
        if len(moves) == 0:
            return 'PASS'
        
        ## Defense from opponent's best move
        
        saveMoves = dict()
        opponentMoves = []
        for i in range(self.goHost.boardSize):
            for j in range(self.goHost.boardSize):
                if self.goHost.board[i][j] == 0:
                    opponentMoves.append((i,j)) 
        
        for oppMove in opponentMoves:
            self.goHost.board[oppMove[0]][oppMove[1]] = 3 - player
            playerDeadPieces = self.goHost.getDeadPieces(player = player)
            self.goHost.board[oppMove[0]][oppMove[1]] = 0
            if len(playerDeadPieces) >= 1:
                saveMoves[oppMove] = len(playerDeadPieces)
                
        sortedSaveMoves = sorted(saveMoves, key = saveMoves.get, reverse = True)
        
        for sm in sortedSaveMoves:
            if sm != None and sm in moves:
                return sm
        
        
        ## Calculating the liberties of the board
        
        opponentLocations = self.getPieceLocations(board = self.goHost.board, player = 3 - player)
        
        ## Getting Neighboring spaces of the opponent's possible moves
        
        emptyXOpponent = []
        neighborsList = []
        for i in opponentLocations:
            neighbors = [(i[0] + operation[0], i[1] + operation[1]) for operation in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if ( (0 <= i[0] + operation[0] < self.goHost.boardSize) and (0 <= i[1] + operation[1] < self.goHost.boardSize))]
            
            for neigh in neighbors:
                neighborsList.append(neigh)
            # neighborsList.extend(neighbors)
                   
        for neigh in neighborsList:
            if self.board[neigh[0]][neigh[1]] == 0:
                emptyXOpponent.append(neigh)
         
        for move in moves:
            tempBoard = deepcopy(self.goHost.board)
            tempBoard[move[0]][move[1]] = player
            deadPieces = self.goHost.getDeadPieces(player = 3 - player, board = tempBoard)
            for deadp in deadPieces:
                tempBoard[deadp[0]][deadp[1]] = 0
            opponentLocations = self.getPieceLocations(board = tempBoard, player = 3 - player)
            emptyYOpponent = []
            
            neighborsList = []
            for i in opponentLocations:
                neighbors = [(i[0] + operation[0], i[1] + operation[1]) for operation in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if ( (0 <= i[0] + operation[0] < self.goHost.boardSize) and (0 <= i[1] + operation[1] < self.goHost.boardSize))]
                
                for neigh in neighbors:
                    neighborsList.append(neigh)
                # neighborsList.extend(neighbors)
                        
            for neigh in neighborsList:
                if self.board[neigh[0]][neigh[1]] == 0:
                    emptyYOpponent.append(neigh)
                    
            if len(emptyXOpponent) - len(emptyYOpponent) >= 1:
                return move
            
        ## Define initial moves to save some time
        
        if len(moves) >= 15:
            if (2,2) in moves:
                return (2,2)
            if (1,1) in moves:
                return (1, 1)
            if (1,3) in moves:
                return (1, 3)
            if (3,1) in moves:
                return (3, 1)
            if (3,3) in moves:
                return (3, 3)
            if (2,0) in moves:
                return (2, 0)
            if (2,4) in moves:
                return (2, 4)
            if (0,2) in moves:
                return (0, 2)
            if (4,2) in moves:
                return (4, 2)
        
        ## Planning the opponent's move in a similar fashion using minimax algorithm
        ## to make our player play wisely
        
        opponentBoard = deepcopy(self.goHost.board)
        opponentPreviousBoard = deepcopy(self.goHost.previousBoard)
        
        oppGoHost = GOHost(player = 3 - player, previousBoard = opponentPreviousBoard, 
                           currentBoard = opponentBoard, n = self.boardSize)
        
        move, score = self.selectOpponentMiniMaxMove(oppGoHost = oppGoHost, player = 3 - player, board = opponentBoard)
        x, y = move[0], move[1]
        
        self.goHost.board[x][y] = 3 - player
        
        freeSpaces = []
        for i in range(self.goHost.boardSize):
            for j in range(self.goHost.boardSize):
                if self.goHost.board[i][j] == 0:
                    freeSpaces.append((i,j)) 
        
        conquests = dict()
        for space in freeSpaces:
            self.goHost.board[space[0]][space[1]] = player
            deadPieces = self.goHost.getDeadPieces(player = 3 - player)
            self.goHost.board[space[0]][space[1]] = 0
            if len(deadPieces) >= 1:
                conquests[space] = len(deadPieces)
                
        sortedConquests = sorted(conquests, key = conquests.get, reverse = True)
        conquestsRemove = []
        
        self.goHost.board[x][y] = 0
        
        if len(sortedConquests) != 0:
            for i in sortedConquests:
                self.goHost.board[i[0]][i[1]] = player ################### ==
                oppMoves = self.getLegalMoves(player = 3 - player, board = self.goHost.board)
                for j in oppMoves:
                    self.goHost.board[j[0]][j[1]] = 3 - player
                    deadPieces = self.goHost.getDeadPieces(player = 3 - player, board = self.goHost.board)
                    self.goHost.board[j[0]][j[1]] = 0
                    if i in deadPieces:
                        conquestsRemove.append(i)
                self.goHost.board[i[0]][i[1]] = 0 
            
            for x in conquestsRemove:
                if x in sortedConquests:
                    sortedConquests.remove(x)
                    
            for i in sortedConquests:
                if i in moves:
                    return i
        
        move, score = self.selectMiniMaxMove(player = player, board = self.goHost.board)
        return move
    
    def play(self, outputPath):
        start = time.time()
        nextStep = self.getNextStep()
        if nextStep == None:
            nextStep = 'PASS'
        end = time.time()
        print("TIME: ", (end-start))
        print("OUTPUT: ", nextStep)
        InputOutputHandler().writeFile(result = nextStep, path = outputPath)


# In[6]:


if __name__ == '__main__':
    boardSize = 5
    inputPath = 'input.txt'
    outputPath = 'output.txt'
    myPlayer, previousBoard, currentBoard = InputOutputHandler().readFile(boardSize, inputPath)
    goHost = GOHost(player = myPlayer, previousBoard = previousBoard, currentBoard = currentBoard, n = boardSize)
    agent = AgentMiniMax(myPlayer, previousBoard, currentBoard, boardSize, goHost)
    agent.play(outputPath)
