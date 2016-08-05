# A function to prompt the user through providing input for a JSON datafile
import solutionTools as st
import inspect
import tools.fileio as io
import ast

print "\nThis function will guide you through creating JSON data for specifying an experiment.\n"

fileName = raw_input("Please enter a file name: ")
dataName = raw_input("Enter a name for the data: ")

data = {"name" : dataName}
print "You will now be prompted to enter a type for the function.  This is used to determine the solver to use.  Possible types are:\n"
funcNames = st.getSolverNames()
for n,t in enumerate(funcNames):
	print "\t{0}.  {1}".format(n,t)
typeNum = int(raw_input("Please enter a type: "))
data['type'] = funcNames[typeNum]

solverArgs = inspect.getargspec(st.solverFunctions[funcNames[typeNum]])[0][1:]
print "\n You will now be guided through entering the necessary arguments for the solver.\n"
for arg in solverArgs:
	data[arg] = ast.literal_eval(raw_input("\t{0} = ".format(arg)))

print "Done with data entry.\nSaving. . ."
io.write_json_params(fileName, data)

