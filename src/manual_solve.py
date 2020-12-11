#!/usr/bin/python

"""
    Andrei Barcovschi, 16451004, Programming and Tools for AI CT5132, Assignment 3

    https://github.com/abarcovschi/ARC
"""

import os, sys
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import colors

def solve_98cf29f8(x):
    """
        This task contains a grid with two shapes of different colours, with a sliver of width 1 cell connected between them.
        The two shapes are of different sizes, with the sliver having the same colour as the smaller shape.
        The transformation involves attaching the smaller shape to the larger shape by following the sliver,
         a sort of rope that someone pulls from the edge of the large shape to bring the small shape to it.

        The solve transformation algorithm involves determining:
            - the colours of the small shape, large shape and sliver;
            - the starting side and ending side of each shape and sliver.
        Then the algorithm copies the large shape into the output grid and based on the location
         of the sliver adds the smaller shape to the correct edge of the larger shape.
        The larger and smaller shapes' sizes are arbitrary and these names are just used to distinguish between them.
        Technically, the algorithm will work even if the "large" shape is smaller than the "small" shape, but the
         grid examples all feature the sliver having the same colour as the smaller shape.

        Status: all training and test grids are solved correctly.
    """
    vertical_transform = False # true if a block of the colour sliver of width 1 is in a row, i.e. sliver is vertical
    # loop over rows to see if sliver is vertical
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size == 1: # the first block of the sliver has been reached
            vertical_transform = True # transformation will be vertical, i.e. colour sliver is vertical
            break
    if vertical_transform: # sliver is vertical
        yhat = core_98cf29f8(x)
    else: # sliver is horizontal
        yhat = (core_98cf29f8(x.T)).T # solve the grid where sliver is vertical by transposing x
    return yhat

def core_98cf29f8(x):
    """core function that does the transformation in vertical direction.
    Transformation code is brought outside of solve_98cf29f8 to reduce clutter as it needs to be called twice.
    Flow control in solve_98cf29f8 calls core_98cf29f8 only once however."""
    colours = set(x[np.nonzero(x)]) # the non black colours in x
    larger_shape_clr_idxs = [] # 3 values: colour of shape, start idx of shape boundary side, end idx of other shape boundary side
    smaller_shape_clr_idxs = [] # 3 values: colour of shape, start idx of shape boundary side, end idx of other shape boundary side
    sliver_clr_idxs = [] # 3 values: colour of the coloured (non black) sliver that connects to the smaller shape, start idx, end idx
    # loop over rows to find the indeces and colour of the sliver's boundaries
    sliver_detected = False
    for i in range(x.shape[0]):
        v = np.nonzero(x[i])[0].size
        if (np.nonzero(x[i])[0].size == 1) and (sliver_detected == False):
            sliver_clr_idxs.append(x[i][np.nonzero(x[i])][0]) # add colour of sliver to sliver_clr_idxs
            sliver_clr_idxs.append(i) # start idx of sliver appended
            sliver_detected = True
        elif (sliver_detected == True) and (np.nonzero(x[i])[0].size != 1):
            sliver_clr_idxs.append(i-1) # end idx of sliver appended
            break
    # loop over rows again to find the indeces and colour of the larger shape's boundaries
    larger_shape_detected = False
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size >= 1:
            # if the colour of the detected shape is not the same as the sliver colour and the larger shape hasn't been detected yet
            if (x[i][np.nonzero(x[i])[0][0]] != sliver_clr_idxs[0]) and (larger_shape_detected == False):
                # the start of the larger shape has been reached
                larger_shape_colour = (colours - set([sliver_clr_idxs[0]])).pop()
                larger_shape_clr_idxs.append(larger_shape_colour) # colour of the larger shape appended
                larger_shape_clr_idxs.append(i) # index of the first row of the larger shape appended
                larger_shape_detected = True
            # else if the colour of the row is the same as the sliver colour and the larger shape was previously detected
            elif (x[i][np.nonzero(x[i])[0][0]] == sliver_clr_idxs[0]) and (larger_shape_detected == True):
                # the end of the larger shape has been reached
                larger_shape_clr_idxs.append(i-1) # the previous row was the last for the larger shape (appended)
                break
        # else if the row is completely black and the larger shape was previously detected
        elif (x[i][np.nonzero(x[i])].size == 0) and (larger_shape_detected == True):
            # the end of the larger shape has been reached
            larger_shape_clr_idxs.append(i-1) # the previous row was the last for the larger shape (appended)
            break
    # loop over rows again to find the indeces of the smaller shape's boundaries
    smaller_shape_detected = False
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

    # start transformations
    yhat = np.zeros(x.shape, dtype=int) # return yhat when transformation is complete
    # add larger shape to yhat
    for i in range(larger_shape_clr_idxs[1], larger_shape_clr_idxs[2]+1):
        yhat[i] = x[i]
    # if sliver is below larger shape
    if larger_shape_clr_idxs[2] < sliver_clr_idxs[2]:
        # draw smaller shape below larger shape
        j = larger_shape_clr_idxs[2] + 1
        for i in range(smaller_shape_clr_idxs[1], smaller_shape_clr_idxs[2]+1):
            yhat[j] = x[i]
            j+=1
    # else the sliver is above larger shape
    else:
        # draw smaller shape above larger shape
        j = larger_shape_clr_idxs[1] - 1
        for i in range(smaller_shape_clr_idxs[1], smaller_shape_clr_idxs[2]+1):
            yhat[j] = x[i]
            j-=1
    return yhat

def solve_4347f46a(x):
    yhat = x.copy()
    for i in range(1, x.shape[0]-1): # loop through rows, i is the x_coordinate of the centre of the 3x3 kernel over the image
        for j in range(1, x.shape[1]-1): # loop through columns of each row, j is the y_coordinate of the centre of the 3x3 kernel over the image
            if x[i][j] != 0: # centre of kernel is over a coloured cell of x
                # convolve the 3x3 kernel with the 3x3 portion of x inside the kernel boundaries
                colour = x[i][j]
                conv_res =   x[i-1][j-1] + x[i-1][j] + x[i-1][j+1] \
                           + x[i][j-1]   + x[i][j]   + x[i][j+1] \
                           + x[i+1][j-1] + x[i][j]   + x[i][j+1]
                if conv_res == colour*9: # maximum convolution result
                    # the centre of kernel is over a coloured cell in x that is inside the boundaries of a coloured shape
                    yhat[i][j] = 0 # change cell inside shape to black in the output grid
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
        show_coloured_result(x, y, yhat) # plot coloured results
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)
        show_coloured_result(x, y, yhat) # plot coloured results

        
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

