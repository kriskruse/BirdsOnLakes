import glob
from bs4 import BeautifulSoup
from collections import Counter
import warnings

def findClasses(paths=None):
    """
    Finds all xml files in path or list of paths, and returns the unique classes and their frequencies.
    This function is highly depended on, that the xml file follows a simple structure. see below
    Expects that the first class name is located at [13] and every class name more than one is 13+n*2 where n is
    the class number

    :param paths: path or list of paths
    :returns: unique_class, occurrences, unique_dict
    """

    if paths is None:
        paths = ["../images/*/*.xml", "../images2/*/*.xml"]
    if type(paths) is not list:
        warnings.warn("Input was not list but has been converted. If this doesn't sound right you might have an issue")
        paths = [paths]
    class_found = []
    occurrences = []

    for path in paths: # loop over paths
        for xmlpath in glob.glob(path): # loop over every element in the path that is a .xml

            # Open and read the xml file
            file = open(xmlpath, 'r')
            data = file.read()
            bs_data = BeautifulSoup(data, "xml")

            # Find and add all the names/classes to the list,
            # we check if the amount of found names/classes follows an expected value
            # The expected value is found with the logic that the data placement is 13, 13+2, 13+2+2, ..
            found = []
            for i in range(13, 25, 2):
                try:
                    class_found.append(bs_data.contents[0].contents[i].contents[1].contents[0])
                    found.append(bs_data.contents[0].contents[i].contents[1].contents[0])
                    first = True
                except:
                    expected = (len(bs_data.contents[0].contents) - 13) / 2
                    if len(found) != expected:
                        print(f"something might be wrong with {xmlpath}")
                    break

    unique_class = list(set(class_found))
    for item in unique_class:
        occurrences.append(class_found.count(item))
    unique_dict = Counter(class_found)
    return unique_class, occurrences, unique_dict


if __name__ == "__main__":
    unique_class, occurrences, unique_dict = findClasses()
    print(f"Total birds {sum(occurrences)}")
    print(unique_class)
    print(occurrences)
    print("")
    print("As a dictionary:")
    print(unique_dict)
