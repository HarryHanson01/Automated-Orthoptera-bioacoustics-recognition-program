import os

def nameFiles(folderPath, species):
    if not os.path.isdir(folderPath):
        print(f"Error: {folderPath} is not a valid directory.")
        return

    files = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
    files.sort()

    count = 1
    for file in files:
        fileExtension = os.path.splitext(file)[1]
        newName = f"{species}_{count}{fileExtension}"   # should probably change the name so 'Test' is not included (for all species)
        newPath = os.path.join(folderPath, newName)

        if os.path.exists(newPath):
            #print(newPath)
            count += 1
            
        else:
            oldPath = os.path.join(folderPath, file)
            os.rename(oldPath, newPath)
            print(f"Renamed: {file} -> {newName}")
            count += 1

folderPath = "FilesStoredForLabelling\\NoID10"  
species = "NoID10"    #make sure to change species tooo
nameFiles(folderPath, species)
