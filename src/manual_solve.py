#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import colors

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_98cf29f8(x):
    colours = set(x[np.nonzero(x)]) # the non black colours in x
    vertical_transform = False # true if a block of the colour sliver of width 1 is in a row
    larger_shape_clr_idxs = [] # 3 values: colour of shape, start idx of shape boundary side, end idx of other shape boundary side
    smaller_shape_clr_idxs = [] # 3 values: colour of shape, start idx of shape boundary side, end idx of other shape boundary side
    sliver_clr_idxs = [] # 3 values: colour of the coloured (non black) sliver that connects to the smaller shape, start idx, end idx
    # loop over rows to first find the sliver colour
    i = 0
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size == 1: # the first block of the colour sliver has been reached
            vertical_transform = True # transformation will be vertical, i.e. colour sliver is vertical
            sliver_clr_idxs.append(x[i][np.nonzero(x[i])][0]) # add colour of sliver to sliver_clr_idxs
            break
    # loop over rows again to find the indeces and colour of the larger shape's boundaries
    larger_shape_detected = False
    i = 0
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size >= 1:
            # if the colour of the detected shape is not the same as the sliver colour and the larger shape hasn't been detected yet
            if (x[i][np.nonzero(x[i])[0][0]] != sliver_clr_idxs[0]) and (larger_shape_detected == False):
                # the start of the larger shape has been reached
                larger_shape_colour = (colours - set([sliver_clr_idxs[0]])).pop()
                larger_shape_clr_idxs.append(larger_shape_colour) # colour of the larger shape appended
                larger_shape_clr_idxs.append(i) # index of the first row of the larger shape appended
                larger_shape_detected = True
            # else if the colour of the row is the same as the sliver colour
            elif (x[i][np.nonzero(x[i])[0][0]] == sliver_clr_idxs[0]) and (larger_shape_detected == True):
                # the end of the larger shape has been reached
                larger_shape_clr_idxs.append(i-1) # the previous row was the last for the larger shape (appended)
                break
        # else if the row is completely black
        elif (x[i][np.nonzero(x[i])].size == 0) and (larger_shape_detected == True):
            # the end of the larger shape has been reached
            larger_shape_clr_idxs.append(i-1) # the previous row was the last for the larger shape (appended)
            break
    # loop over rows again to find the indeces of the smaller shape
    smaller_shape_detected = False
    i = 0
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size > 1:
            # if the colour of the detected shape is the same as the sliver colour and the smaller shape hasn't been detected yet
            if (x[i][np.nonzero(x[i])[0][0]] == sliver_clr_idxs[0]) and (smaller_shape_detected == False):
                # the start of the smaller shape has been reached
                smaller_shape_clr_idxs.append(sliver_clr_idxs[0]) # colour of smaller shape is same as sliver colour
                smaller_shape_clr_idxs.append(i) # index of the first row of the smaller shape appended
                smaller_shape_detected = True
        # else if the number of coloured blocks in the row is less than or equal to 1 (sliver or black row) and smaller shape was previously detected
        elif smaller_shape_detected == True:
            # the end of the smaller shape has been reached
            smaller_shape_clr_idxs.append(i-1) # the previous row was the last for the smaller shape (appended)
            break
    # loop over rows again to find the indeces of the sliver's boundaries
    sliver_detected = False
    i = 0
    for i in range(x.shape[0]):
        v = np.nonzero(x[i])[0].size
        if (np.nonzero(x[i])[0].size == 1) and (sliver_detected == False):
            sliver_clr_idxs.append(i) # start idx of sliver appended
            sliver_detected = True
        elif (sliver_detected == True) and (np.nonzero(x[i])[0].size != 1):
            sliver_clr_idxs.append(i-1) # end idx of sliver appended
            break
    
    yhat = x.copy() # return yhat when transformation is complete



    if vertical_transform == False:
        # loop over columns
        for column in x.T:
            if len(column[np.nonzero(column)]) == 1:
                # start horizontal transformations:
                pass
    return yhat

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
        show_coloured_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)
        show_coloured_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

def show_coloured_result(x, y, yhat):
    """Debug helper function to quickly colour plot the results"""

    cmap = colors.ListedColormap(['black','blue','red','green','yellow', 'grey', 'magenta', 'orange', 'turquoise', 'maroon'])

    # x plot
    f = plt.figure(0)
    ax = f.gca()
    plt.imshow(x, interpolation='nearest', cmap=cmap, vmin=0, vmax=cmap.N)
    plt.title("input")
    plt.tight_layout()
    plt.grid()
    ax.set_xticks(np.arange(x.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(x.shape[0])+0.5, minor=False)

    # y plot
    f = plt.figure(1)
    ax = f.gca()
    plt.imshow(y, interpolation='nearest', cmap=cmap, vmin=0, vmax=cmap.N)
    plt.title("expected output")
    plt.tight_layout()
    plt.grid()
    ax.set_xticks(np.arange(y.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(y.shape[0])+0.5, minor=False)

    # yhat plot
    f = plt.figure(2)
    ax = f.gca()
    plt.imshow(yhat, interpolation='nearest', cmap=cmap, vmin=0, vmax=cmap.N)
    plt.title("created output")
    plt.tight_layout()
    plt.grid()
    ax.set_xticks(np.arange(yhat.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(yhat.shape[0])+0.5, minor=False)
    
    # show all plots
    plt.show()
    
if __name__ == "__main__": main()

