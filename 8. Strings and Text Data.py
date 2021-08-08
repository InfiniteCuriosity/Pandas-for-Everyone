# 8.2 Strings:

'''In Python, a string is simply a series of characters.
They are created by a set of opening and matching single or double quotes. For example,'''

word = 'grail'
sent = 'a scratch'
print(word, sent)

# 8.2.1 Subsetting and Slicing Strings

# 8.2.1.1 get the first letter:

print(word[0])
print(sent[0])

# 8.2.1.2, slicing multiple letters

# get the first three characters

word[:3]

# to get the last three letters:
word[-4:-1]
word[0:5]

# 8.2.1.2, Slicing Multiple Letters

print(word[0:3]) # prints the first three letters of the string

print(sent[-1]) # prints the last letter

print(word[-3:len(word)]) # prints the last three letters, regardless of how long the word is - nice!

# 8.2.2 Getting the Last Character in a String

# note tht the last index i one position smaller than the number returned for len

s_len = len(sent)
print(s_len)

print(sent[2:s_len]) # that works!

# 8.2.2.1 Slicing From the Beginning or to the End

print(word[0:3]) # prints the first three letters

print(sent[2:len(sent)])

# 8.2.2.2 Slicing Increments

# For example, you can get every other string by passing in 2 for every second character.

print(sent[::2]) # prints every other character

print(sent[::3]) # prints every third character

# 8.3 String Methods

word.startswith('g') # true
word.isalnum() # true (is alpha numeric)
word.lower() # that works!
word.strip()
word.center(4)

# 8.4 More String Methods
# 8.4.1 Join
'''The join method takes a container (e.g., a list) and returns a new string containing each element in the list. '''

d1 = '40º'
m1 = "46'"
s1 = '52.837"'
u1 = 'N'

d2 = '73º'
m2 = "58'"
s2 = '26.302"'
u2 = 'W'

coords = ' '.join([d1, m1, s1, u1, d2, m2, s2, u2])
print(coords)

# 8.4.2 Splitlines

multi_str = """Guard: What? Ridden on a horse?
 King Arthur: Yes!
 Guard: You're using coconuts!
 King Arthur: What?
Guard: You've got ... coconut[s] and you're bangin' 'em together. """

multi_str_split = multi_str.splitlines()
print(multi_str_split)

'''Finally, suppose we just wanted the text from the Guard. This is a two-person conversation, so the Guard speaks every other line.'''

guard = multi_str_split[::2]
print(guard) # that works!!

# 8.5 String Formatting

# 8.5.1 Custom String Formatting

var = 'flesh wound'
s = "It's just a {}!"

print(s.format(var))
print(s.format('scratch'))
print(s.format('doornob'))

# using variables multiple times by index:

s = """Black Knights: 'Tis but a {0}.
King Arthur: A {0}? Your arm's off!"""

print(s.format('scratch'))

# it's possible to give the placeholders a variable:

s = 'Hayden Planetarium Coordinates: {lat}, {lon}'
print(s.format(lat = '40.7815º N', lon = '73.9733º W')) # that works!

# 8.5.3 Formatting numbers

print('some digits of pi: {}'.format(3.14159265359))

print("In 2005, Lu Chao of Chine recited {:,} digits of pi".format(67890)) #Great, this adds commas! Nice!

'''Numbers can be used to perform a calculation and formatted to a certain number of decimal values.
Here we can calculate a proportion and format it into a percentage.'''

print("I remember {0:.4} or {0:.4%} of what Lu Chao recited".format(7/67890))

'''Finally, you can use string formatting to pad a number with zeros, similar to how zfill works on strings.'''
print("My ID number is {0:05d}".format(42))

# 8.6 Regular Expressions (RegEx)

# 8.6.1 Match a Pattern

import re
telenum = '1234567890'

m =re.match(pattern='\d\d\d\d\d\d\d\d\d\d', string = telenum)
print(m)
print(bool(m))

# should print 'match'

if m:
    print('match')
else:
    print('no match')

# get the first index of the string match:

print(m.start())

# get the last index of the string match:

print(m.end())

# get the first and last index of the string match:
print(m.span())

# get the string that matched the pattern

print(m.group())

'''Telephone numbers can be a little more complex than a series of 10 consecutive digits.
Here's another common representation.'''

tele_num_spaces = '123 456 7890'
