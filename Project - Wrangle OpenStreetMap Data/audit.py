import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

#OSMFILE_sample = "Pune.osm"
OSMFILE = "Pune.osm"
regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

expected = ["Pune", "Road", "NR", "Avenue", "Corner", "Shivaji", "Bridge", 
            "Society", "Peth"] #expected names in the dataset

mapping = {"pune": "Pune",
           "poona": "Pune",
           "Puneo": "Pune",
           "bridge": "Bridge",
           "road": "Road",
           "Rd": "Road",
           "Rd.": "Road",
           "rasta": "Road",
           "Roads": "Road",
           "path": "Road",
           "marg": "Road",
           "Path": "Road",
           "Marg": "Road",
           "society": "Society",
           "soc.": "Society",
           "Soc.": "Society",
           "Socity": "Society",
           "nagar": "Nagar"
            }

# Search string for the regex. If it is matched and not in the expected list then add this as a key to the set.
def audit_street(street_types, street_name): 
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem): # Check if it is a street name
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile): # return the list that satify the above two functions
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])

    return street_types

#pprint.pprint(dict(audit(OSMFILE_sample))) # print the existing names
pprint.pprint(dict(audit(OSMFILE))) # print the existing names

def string_case(s): # change string into titleCase except for UpperCase
    if s.isupper():
        return s
    else:
        return s.title()

# return the updated names
def update_name(name, mapping):
    name = name.split(' ')
    for i in range(len(name)):
        if name[i] in mapping:
            name[i] = mapping[name[i]]
            name[i] = string_case(name[i])
        else:
            name[i] = string_case(name[i])
    
    name = ' '.join(name)
   

    return name

#update_street = audit(OSMFILE_sample) 
update_street = audit(OSMFILE) 

# print the updated names
for street_type, ways in update_street.iteritems():
    for name in ways:
        better_name = update_name(name, mapping)
        print name, "=>", better_name  