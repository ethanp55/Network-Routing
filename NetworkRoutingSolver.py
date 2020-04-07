#!/usr/bin/python3

from CS312Graph import *
import time
import math

class NetworkRoutingSolver:
    def __init__( self ):
        # List of node distances
        self.node_distances = []
        # List of previous nodes
        self.previous_nodes = []
        # Node map that maps a node to its index in node_distances and previous_nodes
        self.node_map = {}

    def initializeNetwork( self, network ):
        assert( type(network) == CS312Graph )
        self.network = network

    # Function for extracting the path between the source and destination nodes after Dijkstra's algorithm has been
    # run
    # Time complexity is O(|V|)
    # Space complexity is O(|V|)
    def getShortestPath( self, destIndex ):
        self.dest = destIndex

        # Get the source and destination node
        source_node = self.network.nodes[self.source]
        dest_node = self.network.nodes[self.dest]

        # Get the total length of the path from the source to the destination nodes
        total_length = self.node_distances[self.node_map.get(dest_node)]

        # Initialize an empty list of edges
        path_edges = []

        # If the total length is infinity, that means there is no valid path from the source node to the destination
        # node
        if total_length == math.inf:
            return {"cost": total_length, "path": path_edges}

        # Otherwise, there is a valid path and we need to send it to the GUI
        else:
            # Start at the destination node
            curr_node = dest_node

            # Work backwards until we get to the source node
            # Get all of the edges of the path
            # Time complexity of O(|V|) since, in the worst case, our path would contain every node in the graph and we
            # need to iterate through every node in the path in order to send the path to the GUI
            # Space complexity is O(|V|) since, in the worst case, our path would contain every node in the graph and we
            # are storing the path in a list of edges
            while curr_node != source_node:
                curr_node_index = self.node_map.get(curr_node)

                previous_node = self.previous_nodes[curr_node_index][0]
                edge = self.previous_nodes[curr_node_index][1]

                path_edges.append((edge.src.loc, edge.dest.loc, "{:.0f}".format(edge.length)))

                curr_node = previous_node

            # Send the total length and all the edges of the path to the GUI
            return {"cost": total_length, "path": path_edges}

    # Function that computes the path between the source and destination nodes using Dijkstra's algorithm (and either
    # an unsorted array/list or a min heap)
    # If using a min heap, the time complexity is O(|V|log(|V|)).  Otherwise (when using an unsorted array/list), the
    # time complexity is O(|V|^2)
    # With either implementation, the space complexity is O(|V|)
    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time()

        # Clear our global variables that we will use when implementing Dijkstra's algorithm
        self.node_distances.clear()
        self.previous_nodes.clear()
        self.node_map.clear()

        # Run Dijkstra's algorithm with a min heap if the user indicates so
        # Time complexity is O(|V|log(|V|))
        # Space complexity is O(|V|)
        if use_heap:
            self.heap_map = {}
            self.dijstra_with_heap()

        # Otherwise, run Dijkstra's algorithm with an unsorted array/list
        # Time complexity is O(|V|^2)
        # Space complexity is O(|V|)
        else:
            self.dijkstra_with_array()

        t2 = time.time()
        print(t2-t1)
        return (t2-t1)

    # Helper function that runs Dijkstra's algorithm with a min heap
    # Time complexity is O(|V|log(|V|))
    # Space complexity is O(|V|)
    def dijstra_with_heap(self):
        # Initialize the heap (represented in a list)
        heap_array_queue = []

        # Get the source node
        source_node = self.network.nodes[self.source]

        # Iterate through every node and assign their distances to infinity and previous nodes to None (one exception:
        # the source node is initialized with a distance of 0)
        # Time complexity is O(|V|log(|V|)) since we iterate through |V| nodes, and at each iteration we call
        # heap_insert, which has a time complexity of O(log(|V|))
        # Space complexity is O(|V|) since we are storing |V| nodes in the heap, distances list, previous list, and
        # node map
        for curr_node in self.network.nodes:
            if curr_node == source_node:
                self.node_distances.append(0)

            else:
                self.node_distances.append(math.inf)

            self.previous_nodes.append(None)

            # Build our node map as we iterate through each node
            index = len(self.node_distances) - 1
            self.node_map[curr_node] = index

            # Insert each node into our heap
            # Time complexity of O(log(|V|))
            self.heap_insert(heap_array_queue, curr_node)

        # Iterate through the heap until it is empty
        # Time complexity of O(|V|log(|V|)) since, in the worst case, we would iterate through every node in the heap,
        # which has |V| nodes, and at each iteration we call heap_delete_min, which has a time complexity of O(log(|V|))
        while len(heap_array_queue) != 0:
            # Find the node from our heap with the smallest distance (the top node)
            # Time complexity of O(log(|V|))
            smallest_node = self.heap_delete_min(heap_array_queue)

            # Break out of our loop (finish the algorithm) if no smallest node can be found from the heap
            if smallest_node is None:
                break

            # Iterate through each edge coming from the smallest node we found
            # This would have a total time complexity of O(|V|) (not O(|V|) at each iteration) since we will only check
            # a given edge once
            for curr_edge in smallest_node.neighbors:
                # Get the neighbor node from the edge
                neighbor_node = curr_edge.dest

                # Using our node map and list of node distances, find the distances of the smallest node and the
                # neighbor node
                smallest_node_index = self.node_map.get(smallest_node)
                neighbor_node_index = self.node_map.get(neighbor_node)

                smallest_node_distance = self.node_distances[smallest_node_index]
                neighbor_node_distance = self.node_distances[neighbor_node_index]

                # If the sum of the smallest node distance and edge length is less than the distance of the neighbor
                # node, update the distance of the neighbor node and the heap
                if (smallest_node_distance + curr_edge.length) < neighbor_node_distance:
                    self.node_distances[neighbor_node_index] = smallest_node_distance + curr_edge.length
                    self.previous_nodes[neighbor_node_index] = (smallest_node, curr_edge)

                    # Time complexity of O(log(|V|))
                    self.decrease_key(heap_array_queue, neighbor_node)

    # Helper function for inserting nodes into the min heap
    # Time complexity is O(log(|V|)) because this method calls percolate_up in order to properly insert nodes into the
    # heap, which has a time complexity of O(log(|V|))
    # Space complexity is O(1) since we are not creating a copy of the heap/queue and are only creating a few new
    # variables
    def heap_insert(self, heap_array_queue, node):
        # Add the node to the end of the heap
        heap_array_queue.append(node)

        # Get the index of the node we just added in the heap
        index = len(heap_array_queue) - 1

        # Update our node map
        self.heap_map[node] = index

        # Percolate up until the new node is in its proper place in the heap
        # Time complexity of O(log(|V|))
        self.percolate_up(heap_array_queue, index, node)

    # Helper function for deleting the smallest node from the heap (the top node)
    # Time complexity is O(log(|V|)) because this method calls percolate_down in order to properly get the node with the
    # minimum distance from the heap and then update the heap, which has a time complexity of O(log(|V|))
    # Space complexity is O(1) since we are not creating a copy of the heap/queue and are only creating a few new
    # variables
    def heap_delete_min(self, heap_array_queue):
        # Get the smallest node and its distance
        smallest_node = heap_array_queue[0]
        smallest_node_distance = self.node_distances[self.node_map.get(smallest_node)]

        # If the smallest node's distance is infinity, return None (at this point, our algorithm should finish)
        if smallest_node_distance == math.inf:
            return None

        # Remove the bottom node from the heap
        bottom_node = heap_array_queue.pop(len(heap_array_queue) - 1)

        # If the bottom node is the same as the smallest node (our heap only had 1 node in it), we don't need to
        # percolate down/update the heap (just return the node)
        if bottom_node == smallest_node:
            return bottom_node

        # Otherwise, put the bottom node on the top of the heap
        heap_array_queue[0] = bottom_node

        # Remove the smallest node entry from our heap map
        del self.heap_map[smallest_node]

        # Update the heap map for our bottom node
        self.heap_map[bottom_node] = 0

        # Our bottom node is now at index 0 (the first/top node in the heap)
        index = 0

        # Percolate down until the bottom node is in its proper place in the heap and then return the smallest node
        # Time complexity of O(log(|V|))
        self.percolate_down(heap_array_queue, index, bottom_node)

        return smallest_node

    # Helper function for updating nodes in the heap when their distances change
    # Time complexity is O(log(|V|)) because this method calls percolate_up in order to update the heap, which has a
    # time complexity of O(log(|V|))
    # Space complexity is O(1) since we are not creating a copy of the heap/queue and are only creating a few new
    # variables
    def decrease_key(self, heap_array_queue, node):
        # Use the heap map to get the index of where the node is at in the heap
        index = self.heap_map.get(node)

        # Percolate up until the node is in its proper place in the heap
        # Time complexity of O(log(|V|))
        self.percolate_up(heap_array_queue, index, node)

    # Helper function for percolating up the heap
    # Time complexity is O(log(|V|)) because, in the worst case, at each iteration we are swapping the node with it
    # its parent.  This means that, at each iteration, the size of the heap that we have to deal with is half the size
    # of the heap at the previous iteration.  We are also using a map in order to make node index lookup constant time
    # Space complexity is O(1) since we are not creating a copy of the heap/queue and are only creating a few new
    # variables
    def percolate_up(self, heap_array_queue, index, node):
        # Continue to percolate until the node is in its proper place in the heap
        while 1:
            # If the index is 0, the node doesn't have any parents (since it's the top node), so we don't need to
            # percolate up (break out of the loop)
            if index == 0:
                break

            # Otherwise, get the parent node and distances of the node and its parent node
            parent_node = heap_array_queue[(index - 1) // 2]
            parent_node_distance = self.node_distances[self.node_map.get(parent_node)]
            node_distance = self.node_distances[self.node_map.get(node)]

            # If the distance of the parent node is greater than the distance of the node, we need to update the heap
            if parent_node_distance > node_distance:
                # Swap the node and parent node and update their entries in the heap map
                heap_array_queue[(index - 1) // 2] = node
                self.heap_map[node] = (index - 1) // 2
                heap_array_queue[index] = parent_node
                self.heap_map[parent_node] = index

                # Our node is now at the index that its parent was at
                index = (index - 1) // 2

            else:
                break

    # Helper function for percolating down the heap
    # Time complexity is O(log(|V|)) because, in the worst case, at each iteration we are swapping the node with one of
    # its children.  This means that, at each iteration, the size of the heap that we have to deal with is half the size
    # of the heap at the previous iteration.  We are also using a map in order to make node index lookup constant time
    # Space complexity is O(1) since we are not creating a copy of the heap/queue and are only creating a few new
    # variables
    def percolate_down(self, heap_array_queue, index, node):
        # Continue to percolate until the node is in its proper place in the heap
        while 1:
            # If the node is at the very bottom of the heap, we don't need to percolate down (break out of the loop)
            if index == len(heap_array_queue) - 1:
                break

            # Check if the node has a left and right child
            if (2 * index) + 1 <= len(heap_array_queue) - 1 and (2 * index) + 2 <= len(heap_array_queue) - 1:
                # Get the left and right child nodes
                left_child = heap_array_queue[(2 * index) + 1]
                right_child = heap_array_queue[(2 * index) + 2]

                # Get the distances of the node and its left and right children nodes
                left_child_distance = self.node_distances[self.node_map.get(left_child)]
                right_child_distance = self.node_distances[self.node_map.get(right_child)]
                node_distance = self.node_distances[self.node_map.get(node)]

                # Determine which child distance is the smallest and use that corresponding child node for comparing to
                # the actual node
                if right_child_distance < left_child_distance:
                    child_node = right_child
                    child_node_distance = right_child_distance
                    child_index = (2 * index) + 2

                else:
                    child_node = left_child
                    child_node_distance = left_child_distance
                    child_index = (2 * index) + 1

                # If the distance of the node is greater than the distance of its smallest child node, swap them and
                # update their entries in the heap map
                if node_distance > child_node_distance:
                    heap_array_queue[child_index] = node
                    self.heap_map[node] = child_index
                    heap_array_queue[index] = child_node
                    self.heap_map[child_node] = index

                    # The node is node is now at the index that its smallest child was at
                    index = child_index

                # If the node distance is not greater than the distance of its smallest child node, we are done
                # percolating down (break out of the loop)
                else:
                    break

            # If the node doesn't have 2 children, check if it has a left child
            elif (2 * index) + 1 <= len(heap_array_queue) - 1:
                # Get the left child and its index
                left_child = heap_array_queue[(2 * index) + 1]
                left_child_index = (2 * index) + 1

                # Get the distances of the node and its left child
                left_child_distance = self.node_distances[self.node_map.get(left_child)]
                node_distance = self.node_distances[self.node_map.get(node)]

                # If the distance of the node is greater than the distance of its left child, swap them and update
                # their entries in the heap map
                if node_distance > left_child_distance:
                    heap_array_queue[left_child_index] = node
                    self.heap_map[node] = left_child_index
                    heap_array_queue[index] = left_child
                    self.heap_map[left_child] = index

                    # The node is now at the index that its left child was at
                    index = left_child_index

                # If the node distance is not greater than the distance of its left child, we are done percolating down
                # (break out of the loop)
                else:
                    break

            # If the node doesn't have 2 children, check if it has a right child
            elif (2 * index) + 2 <= len(heap_array_queue) - 1:
                # Get the right child and its index
                right_child = heap_array_queue[(2 * index) + 2]
                right_child_index = (2 * index) + 2

                # Get the distances of the node and its right child
                right_child_distance = self.node_distances[self.node_map.get(right_child)]
                node_distance = self.node_distances[self.node_map.get(node)]

                # If the distance of the node is greater than the distance of its right child, swap them and update
                # their entries in the heap map
                if node_distance > right_child_distance:
                    heap_array_queue[right_child_index] = node
                    self.heap_map[node] = right_child_index
                    heap_array_queue[index] = right_child
                    self.heap_map[right_child] = index

                    # The node is now at the index that its right child was at
                    index = right_child_index

                # If the node distance is not greater than the distance of its right child, we are done percolating down
                # (break out of the loop)
                else:
                    break

            # If we arrive here then the node doesn't have any children, which means we are done percolating down (break
            # out of the loop)
            else:
                break

    # Helper function that runs Dijkstra's algorithm with an unsorted array/list
    # Time complexity is O(|V|^2)
    # Space complexity is O(|V|)
    def dijkstra_with_array(self):
        # Initialize the unsorted list
        array_queue = []

        # Get the source node
        source_node = self.network.nodes[self.source]

        # Iterate through every node in the graph and set their distances to infinity and previous nodes to None (except
        # set the distance of the source node to 0)
        # O(|V|) time and space complexity since we are storing |V| nodes in the queue, distances list, previous list,
        # and node map and are iterating through every node
        for curr_node in self.network.nodes:
            if curr_node == source_node:
                self.node_distances.append(0)

            else:
                self.node_distances.append(math.inf)

            self.previous_nodes.append(None)

            index = len(self.node_distances) - 1
            self.node_map[curr_node] = index

            # Add each node to the unsorted list
            array_queue.append(curr_node)

        # Iterate through the unsorted list until there are no more nodes
        # Time complexity is O(|V|^2) because, in the worst case, we would iterate through the entire queue, which has
        # |V| nodes, and at each iteration we call array_delete_min, which has a time complexity of O(|V|)
        while len(array_queue) != 0:
            # Get the smallest node from the unsorted list
            # O(|V|) time complexity
            smallest_node = self.array_delete_min(array_queue)

            # If there is no smallest node, break out of the loop (end the algorithm)
            if smallest_node is None:
                break

            # Iterate through all of the smallest node's edges
            # This would have a total time complexity of O(|V|) (not O(|V|) at each iteration) since we will only check
            # a given edge once
            for curr_edge in smallest_node.neighbors:
                # Get the neighbor node from the edge
                neighbor_node = curr_edge.dest

                # Get the distances of the smallest node and its neighbor node
                smallest_node_index = self.node_map.get(smallest_node)
                neighbor_node_index = self.node_map.get(neighbor_node)

                smallest_node_distance = self.node_distances[smallest_node_index]
                neighbor_node_distance = self.node_distances[neighbor_node_index]

                # If the sum of the smallest node's distance and the edge's length are less than the neighbor node's
                # distance, update the neighbor node's distance
                if (smallest_node_distance + curr_edge.length) < neighbor_node_distance:
                    self.node_distances[neighbor_node_index] = smallest_node_distance + curr_edge.length
                    self.previous_nodes[neighbor_node_index] = (smallest_node, curr_edge)

    # Helper function for deleting the smallest node from the unsorted list and returning it
    # Time complexity is O(|V|) since, in the worst case, we would have to iterate through the entire queue to find the
    # node with the smallest distance
    # Space complexity is O(1) since we are not creating a copy of the array/queue and are only creating a few new
    # variables
    def array_delete_min(self, array_queue):
        # Initialize the minimum distance to infinity and the smallest node to none
        min_distance = math.inf
        smallest_node = None

        # Iterate through each node in the unsorted list
        # O(|V|) time and space complexity, since we have a queue of |V| nodes and we might have to iterate through
        # each node
        for curr_node in array_queue:
            # Get the distance of the current node
            distance = self.node_distances[self.node_map.get(curr_node)]

            # If the distance is less than our current minimum distance, update the current minimum distance and our
            # smallest node
            if distance < min_distance:
                min_distance = distance
                smallest_node = curr_node

        # If we actually found a smallest node, remove it from the unsorted list
        if smallest_node is not None:
            array_queue.remove(smallest_node)

        # Return the smallest node
        return smallest_node
