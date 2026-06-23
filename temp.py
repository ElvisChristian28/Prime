def findMissingAndRepeatedValues(grid:list[int]):
        """
        :type grid: List[List[int]]
        :rtype: List[int]
        """
        dict1 = {}
        output = [0, 0]
        max = grid[0][0]
        min = grid[0][0]

        print(max, min)
        for i in grid:
            for j in i:
                if j not in dict1:
                    dict1[j] = 1
                elif j in dict1:
                    dict1[j] += 1
                
                if j > max:
                    max = j
                if j < min:
                    min = j
        
        for key in dict1:
            if dict1[key] == 2:
                output[0] == dict1[key]
        dict1 = sorted(dict1)
        rng = [i for i in range(min, max + 1)]
        for i in rng:
            if i not in dict1:
                output[1] == i

        print(dict1, max, min, output)

grid = [[9,1,7],[8,9,2],[3,4,6]]
findMissingAndRepeatedValues(grid)
