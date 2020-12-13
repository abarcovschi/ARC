#!/usr/bin/python

"""
    Andrei Barcovschi, 16451004, Programming and Tools for AI CT5132, Assignment 3

    https://github.com/abarcovschi/ARC

    Reflection:

        All of the four tasks were solved using pure numpy and the output grid (yhat) was plotted using matplotlib (show_coloured_result).
        I have tried to use computer vision theory in the implementation of solve_4347f46a and solve_5c2c9af4, which has simplified the task
         of orientation awareness in the 2D world of the input grid, i.e. allowing the algorithm to see locations of colours in the grid more
         effectively.

        All of the algorithms heavily rely on finding the dimensions of the shapes inside the input grid, usually by finding the two diagonally
         opposite corners of square shapes. This can result in the usage of a lot of for loops as the input grid needs to be scanned line by line
         (row by row). It is easier to loop over rows in a numpy 2d array than columns, but if looping over columnds was required, it was easier
         to trasnpose the array and loop over the rows and then transpose the array back when returning. Using numpy to determine whether a row
         has coloured cells or not, something that was used extensively throughout the assignment, turned out to be fairly simple by using the function
         nonzero, something that saved me a lot of manual labour and proved to be extremely helpful in determining the colours used in the input grid.

        Task solve_4347f46a was the only one that could be solved in one iteration over the input image as it was the perfect case for applying
         a true convolution over the entire input grid and applying simple additional logic creating the needed transformations on the fly, resulting
         in the output grid after convolving the entire input grid once. The other tasks required firstly iterating over the entire input grid
         row by row to just find the necessary spacial information of the location of the important coloured cells and only then could transformations
         be applied, which again needed more iterating over the input image. Some of the logic began getting increasingly complex in order to take
         into account all the possible corner cases to make the algorithms as generic as possible, with solve_5c2c9af4 achieving a high level of
         modularity by splitting the task up into three functions and using recursion, something which could not be employed to the same extent for the
         other tasks, especially solve_98cf29f8, which turned out to be rather cumbesome and long.

         Overall, all of these tasks could be solved relatively efficiently solely manipulating numpy arrays and emplying standard algorithmic logic.
"""

import os
import re
import sys
import json
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

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

        Comments: - slightly inefficient as need to loop over rows in x at least 4 times to extract shape information.
                  - transformation is straightforward as operation involves copying entire rows instead of x and y indexing.
                  - algorithm assumes there are only two colours other than black.
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
    """
        Core function that does the transformation in vertical direction.
        Transformation code is brought outside of solve_98cf29f8 to reduce clutter as it needs to be called twice.
        Flow control in solve_98cf29f8 calls core_98cf29f8 only once however.
    """
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
    """
        This task consists of multiple coloured shapes in a grid that need to have their centres hollowed out,
         leaving only the boundaries of the shapes coloured.

        A convolution approach is used to implement this transformation.
        A 3x3 kernel with all values=1 and stride=1 is used iteratively across the input grid, performing a convolution
         with corresponding cells in the input grid that fall in the bounds of the kernel.
        The kernel starts in the top left corner and finishes in the bottom right corner of the input grid.
        Anytime the convolution results in a maximum value of colour*9, this means that all neighbours in all directions
         of a cell in the input grid are also coloured cells and therefore the cell in the input grid indexed by the centre
         of the kernel is inside a coloured shape. The corresponding cell in the output grid is overwritten with 0 as a result.

        Status: all training and test grids are solved correctly.

        Comments: - this algorithm turned out to be very efficient as it could transform on the fly by scanning the input grid
                     row by row, only needing one loop over the input.
                  - indexing in both the row and column directions was needed, resulting in nested for loops.
                  - convolution is a very applicable approach as it is commonly used in computer vision tasks.
                  - algorithm assumes no shapes have boundaries touching the extremities of the input grid.
    """
    yhat = x.copy() # return yhat after transformation is complete
    for i in range(1, x.shape[0]-1): # loop through rows in x, i is the x_coordinate of the centre of the 3x3 kernel over the image
        for j in range(1, x.shape[1]-1): # loop through columns of each row, j is the y_coordinate of the centre of the 3x3 kernel over the image
            if x[i][j] != 0: # centre of kernel is over a coloured cell of x
                # convolve the 3x3 kernel with the 3x3 portion of x inside the kernel boundaries
                colour = x[i][j]
                conv_res =   x[i-1][j-1] + x[i-1][j] + x[i-1][j+1] \
                           + x[i][j-1]   + x[i][j]   + x[i][j+1] \
                           + x[i+1][j-1] + x[i][j]   + x[i][j+1]
                if conv_res == colour*9: # maximum convolution result is when all cells in x under kernel are coloured
                    # the centre of kernel is over a coloured cell in x that IS INSIDE the boundaries of a coloured shape
                    yhat[i][j] = 0 # change cell inside shape to black in the output grid
    return yhat

def solve_a61f2674(x):
    """
        This task consists of an input grid of grey vertical bars, like a bar chart.
        The task is to find the highest and lowest bars and colour the highest blue and lowest red and remove the other
         intermediate height bars.
        
        This approach uses a standard for loop over the rows in the transposed input, since it's easier to loop over rows
         than columns of a numpy array.
        For each row that is not completely black, check if it has more coloured (grey) cells than the maximum or less coloured
         cells than the minimum. If so, store the index for the corresponding case and this way the indeces of the highest
         and lowest bars are found.
        Then the coloured (grey) cells at the index of the highest bar are overwritten with blue, and the grey cells at the index
         of the lowest bar are overwritten with red.
        The transpose of the result is returned, to return vertical bars.

        Status: all training and test grids are solved correctly.

        Comments: - only one loop over the rows in x transpose is needed, and two loops over columns thus making the algorithm
                     relatively efficient.
    """
    x_T = x.T # easier to loop through rows than columns, i.e. use horizontal bars
    yhat = np.zeros(x_T.shape, dtype=int) # return yhat when transformation is complete, initialise to transpose of x with all zeros

    # get the indeces of the highest and lowest bars
    max_height = 0 # maximum height of a coloured bar, initialised to 0
    min_height = x_T.shape[0] # minimum height of a coloured bar, initialised to length of row in x_T
    max_height_idx = -1 # index of row with the highest bar
    min_height_idx = -1 # index of row with the lowest bar
    for i in range(x_T.shape[0]): # loop through rows
        bar_length = np.nonzero(x_T[i])[0].shape[0] # length of the coloured bar in this row
        if bar_length > 0: # a row which has at least 1 coloured cell
            if bar_length >= max_height:
                max_height = np.nonzero(x_T[i])[0].shape[0]
                max_height_idx = i
            if bar_length <= min_height:
                min_height = np.nonzero(x_T[i])[0].shape[0]
                min_height_idx = i
    
    # apply transformations
    for j in range(x_T.shape[1]): # loop through columns of max and min bar rows
        if x_T[max_height_idx][j] != 0:
            yhat[max_height_idx][j] = 1 # max height bar is coloured to blue
        if x_T[min_height_idx][j] != 0:
            yhat[min_height_idx][j] = 2 # min height bar is coloured to red
    return yhat.T # transpose back to vertical bars

def solve_5c2c9af4(x):    
    """
        This task's input grid consists of three uniformly coloured cells on a diagonal line, with the diagonal distance
         between the cells being at least one uncoloured (black) cell.
        
        The goal of the task is to draw concentric squares around the central cell. The first concentric square will include the
         two initial coloured cells as corners. The distance between the concentric squares will be equal to the diagonal distance 
         between the original three cells, called the scale. All square sides that fall inside the bounds of the output 
         grid will be drawn, even if a full concentric square cannot be drawn.

        The implementation borrows concepts from computer vision, namely using a variant of kernel convolution.
        The kernel used is convolved in five locations, top right, bottom right, top left, bottom left and middle only.
        The kernel is slid across the input grid with stride=1. This way the location of the central cell and scale are found.
        Then, using recursive calls, the concentric squares are drawn around the central cell, where each recursive call increases
         the size of the square to draw until the square size is increased so much that none of the sides of the square can be drawn.
        The length of a side of the square needs to be calculated, and two horizontal and two vertical lines representing
        sides of the square are attempted to be drawn, with only those falling within the bounds of the output grid actually being drawn.

        Status: all training and test grids are solved correctly.

        Comments: - the use of computer vision theory of kernel convolutions has again made the implementation more applicable in the domain
                     of AI, if ever so slightly.
                  - recursion simplified the control flow of the algorithm and has helped save space, making the algorithm more concise.
    """
    # functions that return x2, y2, xc or yc coordinates, needed for kernel
    get_coord_2 = lambda c, scale: c + 2*scale + 2 # get new x2 or y2 of kernel based on new value of x1 or y1 respectively, and scale
    get_coord_c = lambda c, scale: c + scale + 1 # get the centre x or y coordinate for kernel by specifying x1 or y1
    # initialise kernel
    # kernel which considers coloured cells on a diagonal from top left to bottom right OR top right to bottom left,
    #  thus able to get location of initial square regardless of direction of the three diagonal coloured cells
    # (x1,y1) = top left location of possible coloured cell
    # (x2,y1) = top right location of possible coloured cell
    # (x1,y2) = bottom left location of possible coloured cell
    # (x2,y2) = bottom right location of possible coloured cell
    # (xc,yc) = centre coloured cell
    kernel = { "x1": 0, "y1": 0, "scale": 1} # scale = diagonal distance between centre cell and a cell on the diagonal
    kernel["x2"] = None
    kernel["y2"] = None
    kernel["xc"] = None
    kernel["yc"] = None

    # get colour
    colour = 0
    for i in range(x.shape[0]):
        if np.nonzero(x[i])[0].size == 1:
            colour = x[i][np.nonzero(x[i])[0][0]]

    # get initial square location
    try:
        max_scale = 5
        for scale in range(1, max_scale+1): # loop through the possible scales
            for y1 in range(x.shape[0]-1-2*scale-2): # loop through all the possible locations y1 could be in at a particular scale (row)
                for x1 in range(x.shape[1]-1-2*scale-2): # loop through all the possible locations x1 could be in at a particular scale (column)
                    # update values in kernel
                    kernel["x1"] = x1
                    kernel["y1"] = y1
                    kernel["scale"] = scale
                    kernel["x2"] = get_coord_2(kernel["x1"],kernel["scale"])
                    kernel["y2"] = get_coord_2(kernel["y1"],kernel["scale"])
                    kernel["xc"] = get_coord_c(kernel["x1"],kernel["scale"])
                    kernel["yc"] = get_coord_c(kernel["y1"],kernel["scale"])
                    # calculate "convolution" of kernel and x to get the location of the first square written into the kernel
                    conv_res = x[kernel["y1"]][kernel["x1"]] + \
                               x[kernel["y2"]][kernel["x1"]] + \
                               x[kernel["y1"]][kernel["x2"]] + \
                               x[kernel["y2"]][kernel["x2"]] + \
                               x[kernel["yc"]][kernel["xc"]]
                            
                    if conv_res == 3*colour:
                        raise Exception # initial square location found, break out of all for loops
    except Exception:
        pass # continue algorithm

    # start transformations
    yhat = np.zeros(x.shape, dtype=int) # return yhat when transformation is complete
    yhat[kernel.get("yc")][kernel.get("xc")] = colour # draw central coloured cell on output grid
    square_num = 1 # initialise to 1 to draw the first concentric square first
    yhat = draw_concentric_squares(yhat, (kernel.get("xc"), kernel.get("yc")), kernel.get("scale"), square_num, colour)

    return yhat

def draw_concentric_squares(out_grid, centre_coord, scale, square_num, colour):
    """
        Helper function for solve_5c2c9af4
        It is a recursive algorithm that draws concentric squares around the central location (xc,yc) from kernel
    """
    xc, yc = centre_coord
    out_w = out_grid.shape[1] # width of output grid
    out_h = out_grid.shape[0] # height of output grid
    side_length = 2*square_num + 2*square_num*scale + 1 # the length of a side of the square to draw
    x1 = xc - (side_length-1)//2 # raw x coordinate of top left corner of square to draw
    y1 = yc - (side_length-1)//2 # raw y coordinate of top left corner of square to draw
    x2 = x1 + side_length - 1 # raw x coordinate of bottom right corner of square to draw
    y2 = y1 + side_length - 1 # raw y coordinate of the bottom right corner of square to draw

    # condition to break out of recursion
    if (x1<0) and (y1<0) and (x2>=out_w) and (y2>=out_h):
        return out_grid # both (x1,y1) and (x2,y2) are outside range of out_grid, no more lines can be drawn
        
    # draw horizontal lines
    x1_draw = max(x1, 0) # if raw x1 is outside range of out_grid, i.e negative, use 0
    x2_draw = min(x2, out_w-1) # if raw x2 is outside range of out_grid, i.e. greater than x1+side_length-1, use max x coord in out_grid
    if y1 >= 0:
        out_grid = draw_line(x1_draw, x2_draw, y1, out_grid, colour, False) # draw top horizontal line
    if y2 < out_h:
        out_grid = draw_line(x1_draw, x2_draw, y2, out_grid, colour, False) # draw bottom horizontal line

    # draw vertical lines
    y1_draw = max(y1, 0)  # if raw y1 is outside range of out_grid, i.e negative, use 0
    y2_draw = min(y2, out_h-1)  # if raw y2 is outside range of out_grid, i.e. greater than y1+side_length-1, use max y coord in out_grid
    if x1 >= 0:
        out_grid = draw_line(y1_draw, y2_draw, x1, out_grid, colour, True)  # draw top vertical line
    if x2 < out_w:  
        out_grid = draw_line(y1_draw, y2_draw, x2, out_grid, colour, True)  # draw bottom vertical line
    
    return draw_concentric_squares(out_grid, centre_coord, scale, square_num+1, colour) # recursive call to draw larger squares

def draw_line(a1, a2, b, out_grid, colour, vert):
    """
        Helper function for solve_5c2c9af4, used by draw_concentric_squares
        It draws a horizontal line between two columns with the row specified in a 2d array

        For an effectively vertical line, transpose array to draw in horizontal direction then transpose back to get vertical line
    """
    if vert: # vertical line
        out_grid = out_grid.T
        for i in range(a1, a2+1):
            out_grid[b][i] = colour
        out_grid = out_grid.T
    else: # horizontal line
        for i in range(a1, a2+1):
            out_grid[b][i] = colour
    return out_grid

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
        #show_coloured_result(x, y, yhat) # plot coloured results
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)
        #show_coloured_result(x, y, yhat) # plot coloured results

        
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
    """Helper function to quickly colour plot the results, used for debugging and simplifying verification of output"""

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

