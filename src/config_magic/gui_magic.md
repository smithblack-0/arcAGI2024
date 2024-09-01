Current setup is fine. Next I would like to start defining what can be displayed and modified. Lets talk data structures.

# Hierarchial display system

The hierarchial display system needs to control two primary things.

These are the color of the display, and the contents of the text entries displayed themselves.

## Text representation.

Each entry in the display should indicate the key used to store the entry in a particular datastructure,
and the type of the entry being stored. Nothing will ever be displayed that does not have at least these
two values. It does not, however, focus on displaying the actual value.

Some examples of this pattern are shown below

* Integer in a list at index 0: "index 0 (int)"
* Instanced dict in a dict at key "foo": "key 'foo' (dict)"
* Generic dict in a dict at key "bizz": "key 'biz' (generic_dict)"

## Colors

These hierarchical patterns can have colors associated with them. This might not be possible. IF not, 
GPT, we can substituted. The idea is that by quickly glancing at the hierarchial tree we get an idea
of what things we need to provide to fill in the pattern.

The color red is used for two conditions. These are

* Unfinished leaves: 
	* If a leaf has a defined type, but no defined value, it will be colored red in the display. 
	* Substitution: Add text (unfinished)
* Unfinished datastructure:
	* If a datastructure contains leaves that have not had their values defined, it will also display as red.
	* Substitution: Add text (unfinished)

The color green is used in two places as well, to indicate when conditions indicated above are satisfied.

* Finished leaves: 
	* If a leaf has had it's entry specified, and it is compatible, we make the text green
* Finished datastructure:
	* If all of a datastructures leaves are complete, it is displayed as green

This lets us know at a glance if a section still needs work, or if it is good. Also of note is two additional
colors, which are grey and yellow. Grey:

* Literal Leaves:
	* They are grey, because the user cannot actually specify anything on them. They are literally locked to be that value

And finally yellow:

* Generics:
	* The generic list and generic dict are yellow. Always. 
	* It indicates to the user the list can be edited. 

## Expansion and selection

Datastructures can be expanded, showing their content. Their leaves can be selected,
which will bring them up in the editor for editing. Additionally, generics can ALSO
be selected, in which case you get brought to the generic creation subsection.

## Backend

As far as the backend-frontend integration is concerned