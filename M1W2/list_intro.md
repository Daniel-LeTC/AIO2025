---
id: list_intro
aliases: []
tags: []
---

# List concept
## Definition:
### List is a array of pointers that assign to addresses saved in heap zone, these addresses don't need to be consequenses. It is a high level data structure
#### Array has 2 types static array and dynamic array
   - Static array cannot expand its elements freely
   - Dynamic array each time it expands, it will double its space => **!WARN: This is not strict, can change depends on context**
   - Python uses dynamic array
   - There is 4 bytes or 8 bytes zone if the array element is value/number
   - There is 1 byte zone if the array element is char
#### List can contain many type of elements inside
   - Python list syntax:
   ```python
                    ls_ex = [0.001, 1, [1,2], 'a', 'anything you need']
   ```
#### List is mutable, able to contain duplicate and in ordered by default
### Methods:
  - Create: using Python list syntax
  - Index: `ls_ex[0]` => 0.001, **Python is a 0 index programming languages**
  - Slicing: `ls_ex[start:end:step]`, **based on step, from start till end have to follow the logic sequence from step - default or stated**, otherwise Slicing will return an empty list 
  - Add element/concat: `ls_ex.append('put your value here')` || `ls_ex + ...`
  - Update: `ls_ex[index] = your value here` 
  - Delete:  `ls_ex.remove(value_of_the_list)` => delete first value satisfied,
  - Stack (lifo) and pop: pop **can extract a value from a list**, we will have a shorter list in-place and this method returns an extract value, `ls_ex.pop()` => This will pop in stack rules, `ls_ex.pop(index)` => extract the value in the current list by its index, we will have a shorter list and get a value from pop 
  - Reverse: `ls_ex.reverse()`
  - Count: `ls_ex.count(value_of_the_list)` => Count value you want to check in the list
  - Copy: `new_ls = ls_ex.copy()`
  - Sort: `ls_ex.sort(reverse=True||False)` => Sort the list in ascending or descending order
  > !WARN: most of methods do not return value, almost all of the actions are conducted in-place, please be bewared when assign variable to a object using methods
### Built-in Funtions:
  > !WARN: Some of the built-in functions below can be applied on many types of array in Python not only List 
  - length of list: return the length of the array => `len()`
  - min value of list: return the minimum value => `min()`
  - max value of list: return the max value => `max()`
  - sum value of list: return the total of all elements inside of the list/array => `sum()`
  - reversed list order: return a reversed iterator objects from the list => `reversed()`
  - iterate by key and value of the array data type: loop through the list and consider both index, value of the elements => `enumerate()`
  - sort value of a list by ascending order: return a ascending order list from the current list => `sorted()`
  - pairing 2 array data type with each other: loop through 2 lists/array and return pair of elements with the same index from 2 arrays => zip(list1, list2) **stop when cannot pair anymore**

## Numpy Array in numpy  NOT an expansion and ultilize of List in Python
  - This data type using C and Fortran to optimize the calculation performance
  - Numpy array requires a strict rules for its elements: All the elements have to be the same data types
  - Numpy array locate in a continuous zone and cannot be fragmented like List in Python

