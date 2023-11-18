import numpy as np
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i,num_1 in enumerate(nums):
            for j,num_2 in enumerate(nums):
                if i==j:
                    continue
                else:
                    if num_1+num_2==target:
                        return [i, j]



method = Solution()
nums = [3,3]
target = 6
out = method.twoSum(nums, target)
print(out)

