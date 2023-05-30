import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

#from google.colab.patches import cv2_imshow

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 6
    BOARD_HEIGHT = 8

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0)],
            90: [(0,0), (0,1), (0,2)],
            180: [ (0,0), (1,0), (2,0)],
            270: [ (0,0), (0,1), (0,2)],
        },
        1: { # L
            0: [(0,0), (0,1), (1,1)],
            90: [(0,0), (0,1), (1,0)],
            180: [(1,1), (0,1), (1,0)],
            270: [(0,0), (1,1), (1,0)],
        },

    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


    def __init__(self,rew):
        self.reset()
        self.rotations=[0,90,180,270]
        self.p0={(4,0):(3,0),(4,180):(3,180),(5,0):(3,0),(5,180):(3,180)}
        self.p1={(5,0):(4,0),(5,90):(4,90),(5,180):(4,180),(5,270):(4,270)}
        self.inv_m={0:self.p0,1:self.p1}
        if rew=='Reward_A':
            self.reward_fun=self.reward_function_A
            self.over=self.terminal_f_A
        if rew=='Reward_B':
            self.reward_fun=self.reward_function_B
            self.over=self.terminal_f_B
        if rew=='Reward_C':
            self.reward_fun=self.reward_function_C
            self.over=self.terminal_f_C
        if rew=='Reward_D':
            self.reward_fun=self.reward_function_D
            self.over=self.terminal_f_D
        if rew=='Reward_E':
            self.reward_fun=self.reward_function_E
            self.over=self.terminal_f_E
        if rew=='Reward_F':
            self.reward_fun=self.reward_function_F
            self.over=self.terminal_f_F

    def terminal_f_A(self):
        return -2
    def terminal_f_B(self):
        return -1
    def terminal_f_C(self):
        return -100
    def terminal_f_D(self):
        return -100
    def terminal_f_F(self):
        return -100

    def reward_function_A(self,lines_cleared):
        rew=1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        return rew

    def reward_function_B(self,lines_cleared):
        rew=(lines_cleared *10)
        return rew

    def reward_function_C(self,lines_cleared):
        props=self._get_board_props(self.board)
        lines=lines_cleared
        holes=props[1]
        bumpiness=props[2]
        sum_height=props[3]

        rew=(lines *10)+(-5*holes)+(-2*bumpiness)+(-sum_height)
        return rew

    def reward_function_D(self,lines_cleared):
        props=self._get_board_props(self.board)
        lines=lines_cleared
        holes=props[1]
        bumpiness=props[2]
        sum_height=props[3]

        rew=10*(10**lines)+(-5*holes)+(-2*bumpiness)+(-sum_height)
        return rew

    def reward_function_E(self,lines_cleared):
        props=self._get_board_props_rew(self.board)
        lines=lines_cleared
        holes=props[1]
        bumpiness=props[2]
        sum_height=props[3]
        max_height=props[4]
        rew=(500*lines)+(-10*holes)+(-2*bumpiness)+(-1*sum_height)+(-5*max_height)
        return rew

    def reward_function_F(self,lines_cleared):
        props=self._get_board_props(self.board)
        lines=lines_cleared
        holes=props[1]
        bumpiness=props[2]
        sum_height=props[3]
        max_height=props[4]
        rew=(1000*lines)+(-5*holes)+(-2*bumpiness)+(-2*sum_height)+(-5*max_height)
        return rew


    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score


    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces

        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        # if self._check_collision(self._get_rotated_piece(), self.current_pos):
        #     self.game_over = True
        states=self.get_next_states()
        if len(states)==0 :
              self.game_over=True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)

        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]


    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece

        rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def check_terminal(self,piece_id):
        rotation=self.current_rotation
        piece=Tetris.TETROMINOS[piece_id][rotation]
        min_x = min([p[0] for p in piece])
        max_x = max([p[0] for p in piece])
        for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
            pos = [x, 0]
            while not self._check_collision(piece, pos):
                pos[1] += 1
            pos[1] -= 1
            if pos[1] <0:
                return True
        return False


    def get_piece_actions(self,piece_id):
        '''Get all actions '''
        actions=[]
        rotations = [0, 90, 180, 270]
        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    actions.append((x, rotation))
        return actions
    def get_all_actions(self):
        temp=[]
        for i in range(0,Tetris.BOARD_WIDTH):
            for j in self.rotations:
                temp.append((i,j))
        return temp

    def get_all_actions_old(self):
        temp=[]
        for i in range(0,2):
            temp1=self.get_piece_actions(i)
            for j in temp1:
                temp.append(j)
        actions=set(temp)
        return actions

    def get_next_boards(self):
        '''Get all possible next states'''
        pos_board={}
        piece_id = self.current_piece

        rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    #states[(x, rotation)] = self._get_board_props(board)
                    pos_board[(x, rotation)]=board
        return pos_board

    def get_state_size(self):
        '''Size of the state'''
        return 4


    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        p=self.current_piece
        temp=(x,rotation)
        if temp in self.inv_m[p].keys():
            x,rotation=self.inv_m[p][temp]

        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        # score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        score=self.reward_fun(lines_cleared)
        self.score += score
        if self.check_terminal(p):
            self.game_over=True
        # Start new round
        self._new_round()
        if self.game_over:
            score +=self.over()

        return score, self.game_over,lines_cleared


    def render(self):
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

    def play_g(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over
         with google colab'''

        p=self.current_piece
        temp=(x,rotation)
        if temp in self.inv_m[p].keys():
            x,rotation=self.inv_m[p][temp]
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render_g()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        #score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        score= 1+(lines_cleared*10)
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over,lines_cleared

    def render_g(self):
        '''Renders the current board  with google colab'''
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        #img2=cv2.imread('image', np.array(img))
        cv2_imshow(img)
        cv2.waitKey(1)
        print()
