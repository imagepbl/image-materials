# to be deleted once the 'constants.py' become deprecated. created to keep the same structure as buildings and vehicles, and to add categories not defined elsewhere

# End-of-Life types
EolTypes = prism.Dimension("eol", [
    "reusable",
    "recyclable",
    "losses",
    "surplus losses"
]
)