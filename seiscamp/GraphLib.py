import logging

Logger = logging.getLogger(__name__)


class Graph:
    """
    Class to find connected components of an undirected graph using DFS algorithm
    Example:
    Create a graph given in the above diagram, 5 vertices numbered from 0 to 4

    g = Graph(5)
    g.addEdge(1, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 4)
    cc = g.connectedComponents()
    print("Following are connected components")
    print(cc)
    https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    """

    def __init__(self, V):
        """
        init function to declare class variables
        :param V: Number of distinct values in graph
        """
        self.V = V
        self.adj = [[] for i in range(V)]

    def dfs_util(self, temp, v, visited):
        """
        DFS algorithm to find islands in graph
        :param temp:
        :param v:
        :param visited:
        :return:
        """
        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.dfs_util(temp, i, visited)
        return temp

    def add_edge(self, v, w):
        """
        method to add an undirected edge linking element v to w
        """
        self.adj[v].append(w)
        self.adj[w].append(v)

    def connected_components(self):
        """
        Method to retrieve connected components in an undirected graph
        :return: connected components are list of list
        """
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.dfs_util(temp, v, visited))
        return cc
