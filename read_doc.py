class ReadDoc(object):
    def __init__(self,filepath):
        self.filepath = filepath
        #self.load_data()
    def getConversations(self):
        with open(self.filepath,"r",encoding="utf8") as f:
            lines = f.readlines()
        self.conversion = list(map(lambda x:x.split("\t"),lines[1:]))
        return self.conversion
    def load_data(self):
        with open(self.filepath,"r",encoding="utf8") as f:
            lines = f.readlines()
        list_docs = list(map(lambda x:x.split("\t"),lines[1:]))
        names = []
        category_name = []
        brand_names = []
        desc = []
        for l in list_docs:
            if "&" in l[3]:
                for elem in l[3].split("&"):
                    if "/" in elem:
                        for e in elem.split("/"):
                            category_name.append(e.strip())
                    else:
                        category_name.append(elem.strip())
            elif "/" in l[3]:
                for e in l[3].split("/"):
                    category_name.append(e.strip())
            elif l[3].replace(" ","") is not "":
                category_name.append(l[3].replace(" ",""))
            names.append(l[1].strip())
            if l[4].replace(" ", "") is not "":
                brand_names.append(l[4].strip())
            if l[-1].strip() is not "":
                desc.append(l[-1].strip().replace("\n", ""))
        self.brand_names = list(set(brand_names))
        self.category_names = list(set(category_name))
        self.names = list(set(names))
"""
read_data = ReadDoc("data/train.tsv")
print(read_data.brand_names[:100])
print(len(read_data.brand_names))
print(len(read_data.category_names))
"""