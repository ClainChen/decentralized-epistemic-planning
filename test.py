from collections import Counter

def remove_continue_duplicates(lst):
    if not lst:
        return []
    new_list = [lst[0]]
    for ele in lst:
        if ele != new_list[-1]:
            new_list.append(ele)
    return new_list

lst = [1] *2 + [3,2,4,1,1,6,5,5,5,5,2,2,8,9,10]
print(remove_continue_duplicates(lst))
