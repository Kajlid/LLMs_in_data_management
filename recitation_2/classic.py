import random

x_size = 15
y_size = 15

state_space={}
actions = {}
terminal = {}
value = {}
discount = 0.95

def north(state):
    return (state[0], state[1] - 1)

def south(state):
    return (state[0], state[1] + 1)

def east(state):
    return (state[0]+1, state[1])

def west(state):
    return (state[0]-1, state[1])

def make_world():
    finish = (random.randrange(x_size), random.randrange(y_size))
    fail = (random.randrange(x_size), random.randrange(y_size))
    if finish != fail:
        for x in range(0,x_size):
            for y in range(0,y_size):
                state_space[(x,y)] = 0
                value[(x,y)] = 0
                a = []
                if x != 0:
                    a.append(west)
                if x != x_size-1:
                    a.append(east)
                if y != 0:
                    a.append(north)
                if y != y_size-1:
                    a.append(south)
                actions[(x,y)] = a

        state_space[finish] = 1
        terminal[finish] = True
        terminal[fail] = True
        state_space[fail] = -1
    else:
        make_world()

def show_state(state_space,current_state=None):
    print("-"*(x_size+2))
    for y in range(0, y_size):
        line = "|"
        for x in range(0, x_size):
            if current_state and x==current_state[0] and y==current_state[1]:
                line += "🐈"
            elif state_space[(x, y)] == 1:
                line += "🥛"
            elif state_space[(x, y)] == -1:
                line += "☢"
            else:
                line += " "
        barrier = "🧱"

        print(line + "|")
    print("_" * (x_size+2))

def walk(state_space, show=True, randomized=False):
    current_state = (random.randrange(x_size), random.randrange(y_size))
    trajectory=[current_state]
    reward = 0
    if show:
        show_state(state_space, current_state)
    while current_state not in terminal:
        if randomized:
            act = random.choice(actions[current_state])
        else:
            candidates = []
            possible_actions = actions[current_state]
            possible_actions = random.sample(possible_actions, len(possible_actions))
            for possible_act in possible_actions:
                candidates.append([possible_act,possible_act(current_state)])
            act = sorted(candidates,key=lambda x: -value[x[1]])[0][0]
        current_state = act(current_state)
        trajectory.append(current_state)
        reward += state_space[current_state]
        if show:
            show_state(state_space, current_state)

    dfactor = 1
    for state in reversed(trajectory):
        value[state] = value[state] + reward*dfactor
        dfactor = dfactor * discount
        
    steps = len(trajectory) - 1
    start_state = trajectory[0]
    
    if state_space[current_state] == 1:
        optimal_steps = abs(start_state[0] - current_state[0]) + abs(start_state[1] - current_state[1])
        print("steps:", steps)
        print("optimal steps:", optimal_steps)
        
        return True
    
    return False
    
    

def normalize_values(value):
    max = 0
    for x in range(0, x_size):
        for y in range(0, y_size):
            if abs(value[(x, y)]) > max:
                max = abs(value[(x, y)])
    for x in range(0, x_size):
        for y in range(0, y_size):
            if max>0:
                value[(x,y)] = value[(x,y)]/max

def walks(trials):
    success_count = 0
    for _ in range(trials):
        success = walk(state_space, randomized=True, show=True)
        
        if success:
            success_count += 1
            
    print(f"Success rate: {success_count}/{trials}")

if __name__ == "__main__":
    make_world()
    normalize_values(value)
    # walk(state_space, randomized=False, show=True)
    walks(100)
