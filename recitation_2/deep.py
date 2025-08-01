import torch
import torch.nn as nn
import random
import os

rewards={}
# x_size = 6
# y_size = 6
x_size = 10
y_size = 10
# x_size = 200
# y_size = 200
rewards[(0,0)] = 1
rewards[(5,5)] = -1

barriers = {}
barriers[(3,3)] = True
barriers[(1,0)] = True

def north(state):
    return (state[0], state[1] - 1)

def south(state):
    return (state[0], state[1] + 1)

def east(state):
    return (state[0]+1, state[1])

def west(state):
    return (state[0]-1, state[1])

atoi = {
    'north':0,
    'south':1,
    'east':2,
    'west':3
}

itoa = {
    0: north,
    1: south,
    2: east,
    3: west
}

iopposite ={
    0:1,
    1:0,
    2:3,
    3:2
}


def available_actions(state):
    x=state[0]
    y=state[1]
    a = []
    
    # Make sure that the action would not lead to a barrier
    if x != 0 and (x - 1, y) not in barriers:
        a.append(west)
    if x != x_size-1 and (x + 1, y) not in barriers:
        a.append(east)
    if y != 0 and (x, y - 1) not in barriers:
        a.append(north)
    if y != y_size-1 and (x, y + 1) not in barriers:
        a.append(south)
    return a

def show_state(current_state):
    print("-"*(x_size+2))
    for y in range(0, y_size):
        line = "|"
        for x in range(0, x_size):
            if current_state and x==current_state[0] and y==current_state[1]:
                line += "🐈"
            elif (x,y) in rewards and rewards[(x, y)] == 1:
                line += "🥛"
            elif (x,y) in rewards and rewards[(x, y)] == -1:
                line += "☢"
            elif (x, y) in barriers:  # Display barriers
                line += "🧱"
            else:
                line += " "
        #barrier = "🧱"

        print(line + "|")
    print("_" * (x_size+2))


def make_training_corpus(trials):
    print(f"Building a {trials} Random Walk Corpus ")
    training_corpus = open("training.txt", "w")
    for i in range(trials):
        current_state = (random.randrange(x_size), random.randrange(y_size))
        trajectory = [current_state]
        action_sequence = []
        reward = 0
        while current_state not in rewards:
            act = random.choice(available_actions(current_state))
            current_state = act(current_state)
            action_sequence.append(atoi[act.__name__])
            trajectory.append(current_state)
            if current_state in rewards:
                reward = rewards[current_state]
                
                
        L = len(action_sequence)
        gamma=0.95
        if reward > 0:
            for i in range(max(len(action_sequence)-(x_size+y_size),0), len(action_sequence)):
                discount = gamma ** (L - i - 1)
                training_corpus.write(f"{trajectory[i][0]/x_size},{trajectory[i][1]/y_size},{action_sequence[i]},{discount}\n")
                # training_corpus.write(f"{trajectory[i][0]/x_size},{trajectory[i][1]/y_size},{action_sequence[i]}\n")
        else:
            for i in range(max(len(action_sequence)-2,0), len(action_sequence)):
                discount = gamma ** (L - i - 1)  # number of steps remaining after taking the action i
                training_corpus.write(f"{trajectory[i][0]/x_size},{trajectory[i][1]/y_size},{iopposite[action_sequence[i]]},{discount}\n")
                #training_corpus.write(f"{trajectory[i][0]/x_size},{trajectory[i][1]/y_size},{iopposite[action_sequence[i]]}\n")

    training_corpus.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100

input_size = 2
# hidden_size = 10
hidden_size = 32
num_classes = 4
learning_rate = 0.001

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists("model.pt"):
    print("loading parameters from model.pt file")
    model.load_state_dict(torch.load("model.pt", map_location=device))
else:
    print('Starting from scratch')



def train(num_epochs):
    training_corpus = open("training.txt","r")

    batches = []
    clabels = []
    discount_batch = []
    try:
        i = 0
        batch = []
        label = []
        discounts = []
        while True:
            next_line = training_corpus.readline().split(',')
            x=float(next_line[0])
            y=float(next_line[1])
            a=int(next_line[2].strip())
            discount = float(next_line[3].strip())
            batch.append([x,y])
            label.append(a)
            discount_batch.append(discount)
            i+=1
            if i==batch_size:
                batches.append(torch.FloatTensor(batch))
                clabels.append(torch.LongTensor(label))
                discounts.append(torch.FloatTensor(discount_batch))
                batch = []
                label = []
                discount_batch = []
                i=0
    except:
        pass

    # Train the model
    n_total_steps = len(batches)
    for epoch in range(num_epochs):
        sum_loss = 0
        for i in range(len(batches)):
            positions = batches[i].to(device)
            labels = clabels[i].to(device)
            discount = discounts[i].to(device)
            outputs = model(positions)
            loss = criterion(outputs, labels) 
            loss = (loss * discount).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()

        torch.save(model.state_dict(), "model.pt")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum_loss/len(batches):.4f}')


def deep_walk():
    # start on non-terminal state, avoid infinity recursion
    while True:
        current_state = (random.randrange(x_size), random.randrange(y_size))
        if current_state not in rewards:
            break
    trajectory = [current_state]
    if current_state not in rewards:
        while current_state not in rewards:
            position = torch.FloatTensor([current_state[0]/x_size,current_state[1]/y_size])
            outputs = model(position)
            outputs = outputs/temperature
            outputs = outputs.softmax(0)

            dist = torch.distributions.categorical.Categorical(probs=outputs)
            while True:
                selected = dist.sample()
                act = itoa[selected.item()]
                if act in available_actions(current_state):
                    break
            current_state = act(current_state)
            trajectory.append(current_state)
            show_state(current_state)
    else:
        deep_walk()
        
    steps = len(trajectory) - 1
    start_state = trajectory[0]
    
    if rewards[current_state] == 1:
        optimal_steps = abs(start_state[0] - current_state[0]) + abs(start_state[1] - current_state[1])
        print("steps:", steps)
        print("optimal steps:", optimal_steps)
        
        # if steps <= optimal_steps * 2:
        # print("Good chance of finding milk in close to optimal steps:", bool((steps/optimal_steps) <= 2))
        
        return True
    
    return False


def evaluation(tests=100):
    success_count = 0
    for _ in range(tests):
        success = deep_walk()
        if success:
            success_count += 1
    print(f"Success rate: {success_count}/{tests}")
        

temperature = 1

if __name__ == "__main__":
    make_training_corpus(10000)
    train(100)
    deep_walk()
    evaluation()
