# import xml.etree.ElementTree as ET

# tree = ET.parse("tests/annotation/0.xml")
# root = tree.getroot()
# class_dictionary = {}
# classes = 1
# class_dictionary["background"] = 0
# class_bnd_box = {}
# class_bnd_box["classes"] = {}
# class_bnd_box["size"] = {}
# for child in root:
#     if child.tag == "size":
#         for size in child:
#             class_bnd_box["size"][size.tag] = int(size.text)
#     if child.tag == "object":
#         current_tag = ""
#         for elem in child:
#             bnd_dict = {}
#             if elem.tag == "name":
#                 if elem.text not in class_dictionary.keys():
#                     class_dictionary[elem.text] = classes
#                     classes += 1
#                 if elem.text not in class_bnd_box["classes"].keys():
#                     class_bnd_box["classes"][elem.text] = []
#                 current_tag = elem.text
#             if elem.tag == "bndbox":
#                 for coord in elem:
#                     bnd_dict[coord.tag] = float(coord.text)
#                 class_bnd_box["classes"][current_tag].append(bnd_dict)

# print(class_dictionary)
# print(class_bnd_box)


class DataStructure:
    num : int
    string : str

v = DataStructure()
v.num = 10
v.string = "asdf"


print(v.num)