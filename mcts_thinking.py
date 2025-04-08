#!/usr/bin/env python3
"""
MCTS Sequential Thinking Server (MCP SDK Version)
------------------------------------------------
A Monte Carlo Tree Search enhanced sequential thinking server for improving
LLM reasoning, implemented using the MCP Python SDK for Claude Desktop integration.
"""

import math
import logging
import random
import time
from typing import Dict, List, Any, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCTSThinkingManager:
    """Monte Carlo Tree Search enhanced sequential thinking manager."""
    
    def __init__(self):
        self.thought_history = []
        self.node_map = {}
        self.root_nodes = []
        self.exploration_constant = 1.414  # Default UCB1 exploration constant (sqrt(2))
    
    def generate_node_id(self) -> str:
        """Generate a unique node ID."""
        return f"node_{int(time.time() * 1000)}_{random.randint(0, 10000)}"
    
    def validate_thought_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize thought data."""
        # Validate required fields
        if 'thought' not in input_data or not isinstance(input_data['thought'], str):
            raise ValueError('Invalid thought: must be a string')
        
        if 'thoughtNumber' not in input_data or not isinstance(input_data['thoughtNumber'], int):
            raise ValueError('Invalid thoughtNumber: must be an integer')
        
        if 'totalThoughts' not in input_data or not isinstance(input_data['totalThoughts'], int):
            raise ValueError('Invalid totalThoughts: must be an integer')
        
        if 'nextThoughtNeeded' not in input_data or not isinstance(input_data['nextThoughtNeeded'], bool):
            raise ValueError('Invalid nextThoughtNeeded: must be a boolean')
        
        # Handle optional MCTS-specific fields
        node_id = input_data.get('nodeId') if isinstance(input_data.get('nodeId'), str) else self.generate_node_id()
        parent_id = input_data.get('parentId') if isinstance(input_data.get('parentId'), str) else None
        
        visits = input_data.get('visits') if isinstance(input_data.get('visits'), int) else 1
        value_estimate = input_data.get('valueEstimate') if isinstance(input_data.get('valueEstimate'), float) else 0.5
        
        child_nodes = input_data.get('childNodes', [])
        if not isinstance(child_nodes, list):
            child_nodes = []
        
        # Calculate depth automatically if not provided
        depth = input_data.get('depth')
        if not isinstance(depth, int):
            if parent_id and parent_id in self.node_map:
                depth = self.node_map[parent_id].get('depth', 0) + 1
            else:
                depth = 0
        
        action = input_data.get('action') if isinstance(input_data.get('action'), str) else None
        exploration_constant = input_data.get('explorationConstant') if isinstance(input_data.get('explorationConstant'), float) else self.exploration_constant
        
        return {
            # Original sequential thinking fields
            'thought': input_data['thought'],
            'thoughtNumber': input_data['thoughtNumber'],
            'totalThoughts': input_data['totalThoughts'],
            'nextThoughtNeeded': input_data['nextThoughtNeeded'],
            'isRevision': input_data.get('isRevision'),
            'revisesThought': input_data.get('revisesThought'),
            'branchFromThought': input_data.get('branchFromThought'),
            'branchId': input_data.get('branchId'),
            'needsMoreThoughts': input_data.get('needsMoreThoughts'),
            
            # MCTS fields
            'nodeId': node_id,
            'parentId': parent_id,
            'visits': visits,
            'valueEstimate': value_estimate,
            'childNodes': child_nodes,
            'depth': depth,
            'action': action,
            'explorationConstant': exploration_constant
        }
    
    def calculate_ucb(self, node: Dict[str, Any]) -> float:
        """Calculate Upper Confidence Bound (UCB1) score for a node."""
        # If no visits, treat as infinity (ensure exploration)
        if node['visits'] == 0:
            return float('inf')
        
        parent_node = self.node_map.get(node['parentId']) if node.get('parentId') else None
        parent_visits = parent_node['visits'] if parent_node else node['visits']
        
        # UCB1 formula: value + C * sqrt(ln(parentVisits) / visits)
        exploration_term = math.sqrt(math.log(parent_visits) / node['visits'])
        return node['valueEstimate'] + (node.get('explorationConstant', self.exploration_constant) * exploration_term)
    
    def get_recommended_nodes(self) -> List[Dict[str, Any]]:
        """Get node recommendations based on UCB scores."""
        # Get all leaf nodes that need more thinking
        leaf_nodes = [
            node for node in self.node_map.values() 
            if len(node['childNodes']) == 0 and node['nextThoughtNeeded']
        ]
        
        # Calculate UCB score for each
        scored_nodes = [
            {
                'nodeId': node['nodeId'],
                'ucbScore': self.calculate_ucb(node),
                'thought': (node['thought'][:50] + '...') if len(node['thought']) > 50 else node['thought']
            } 
            for node in leaf_nodes
        ]
        
        # Sort by UCB score (descending)
        return sorted(scored_nodes, key=lambda x: x['ucbScore'], reverse=True)
    
    def update_node_value(self, node_id: str, new_value: float) -> None:
        """Update node value and propagate changes upward (backpropagation)."""
        node = self.node_map.get(node_id)
        if not node:
            return
        
        # Update this node's value estimate with the new sample
        node['visits'] += 1
        node['valueEstimate'] = ((node['visits'] - 1) * node['valueEstimate'] + new_value) / node['visits']
        
        # Recursively update parent nodes (backpropagation)
        if node.get('parentId'):
            self.update_node_value(node['parentId'], new_value)
    
    def format_thought(self, thought_data: Dict[str, Any]) -> str:
        """Format thought data for display."""
        prefix = ''
        context = ''
        
        if thought_data.get('isRevision'):
            prefix = '🔄 Revision'
            context = f" (revising thought {thought_data.get('revisesThought')})"
        elif thought_data.get('branchFromThought'):
            prefix = '🌿 Branch'
            context = f" (from thought {thought_data.get('branchFromThought')}, ID: {thought_data.get('branchId')})"
        else:
            prefix = '💭 Thought'
            context = ''
        
        header = f"{prefix} {thought_data['thoughtNumber']}/{thought_data['totalThoughts']}{context}"
        
        node_id_short = thought_data['nodeId'][:8] + '...' if len(thought_data['nodeId']) > 8 else thought_data['nodeId']
        parent_id_short = (thought_data['parentId'][:8] + '...') if thought_data.get('parentId') and len(thought_data['parentId']) > 8 else 'root'
        
        mcts_info = f"Node: {node_id_short} | Parent: {parent_id_short} | Visits: {thought_data['visits']} | " \
                    f"Value: {thought_data['valueEstimate']:.3f} | Depth: {thought_data['depth']}"
        
        # Calculate border width
        max_width = max(len(header), len(mcts_info), len(thought_data['thought']))
        border = '─' * (max_width + 4)
        
        # Format with box drawing characters
        lines = [
            f"┌{border}┐",
            f"│ {header.ljust(max_width + 2)} │",
            f"│ {mcts_info.ljust(max_width + 2)} │",
            f"├{border}┤",
            f"│ {thought_data['thought'].ljust(max_width + 2)} │",
            f"└{border}┘"
        ]
        
        return '\n'.join(lines)
    
    def process_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a thought and return the response."""
        try:
            validated_input = self.validate_thought_data(input_data)
            
            # Adjust total thoughts if needed
            if validated_input['thoughtNumber'] > validated_input['totalThoughts']:
                validated_input['totalThoughts'] = validated_input['thoughtNumber']
            
            # Add to node map
            self.node_map[validated_input['nodeId']] = validated_input
            
            # If this node has a parent, add it to the parent's children
            if validated_input.get('parentId') and validated_input['parentId'] in self.node_map:
                parent = self.node_map[validated_input['parentId']]
                if validated_input['nodeId'] not in parent['childNodes']:
                    parent['childNodes'].append(validated_input['nodeId'])
            else:
                # This is a root node
                if validated_input['nodeId'] not in self.root_nodes:
                    self.root_nodes.append(validated_input['nodeId'])
            
            # Add to linear history
            self.thought_history.append(validated_input)
            
            # Format and display
            formatted_thought = self.format_thought(validated_input)
            logger.info(formatted_thought)
            
            # Get node recommendations based on UCB scores
            recommendations = self.get_recommended_nodes()[:3]
            
            # Calculate tree statistics
            all_nodes = list(self.node_map.values())
            tree_stats = {
                'nodeCount': len(all_nodes),
                'maxDepth': max([n['depth'] for n in all_nodes]) if all_nodes else 0,
                'averageValue': sum([n['valueEstimate'] for n in all_nodes]) / len(all_nodes) if all_nodes else 0
            }
            
            return {
                # Original sequential thinking info
                'thoughtNumber': validated_input['thoughtNumber'],
                'totalThoughts': validated_input['totalThoughts'],
                'nextThoughtNeeded': validated_input['nextThoughtNeeded'],
                'thoughtHistoryLength': len(self.thought_history),
                
                # MCTS specific info
                'currentNodeId': validated_input['nodeId'],
                'currentValue': validated_input['valueEstimate'],
                'visits': validated_input['visits'],
                'treeStats': tree_stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error processing thought: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'status': 'failed'
            }

# Create an MCP server instance
mcp = FastMCP("MCTS Thinking")

# Create a singleton MCTS thinking manager
thinking_manager = MCTSThinkingManager()

@mcp.tool()
def mctsthinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int, 
    nextThoughtNeeded: bool,
    nodeId: Optional[str] = None,
    parentId: Optional[str] = None,
    visits: Optional[int] = None,
    valueEstimate: Optional[float] = None,
    childNodes: Optional[List[str]] = None,
    depth: Optional[int] = None,
    action: Optional[str] = None,
    explorationConstant: Optional[float] = None,
    isRevision: Optional[bool] = None,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: Optional[bool] = None
) -> dict:
    """
    A Monte Carlo Tree Search (MCTS) enhanced tool for dynamic and reflective problem-solving.
    This tool helps analyze problems through tree search with value estimation similar to AlphaZero.
    Each thought is a node in a search tree, with exploration and exploitation balanced via UCB1.

    When to use this tool:
    - Breaking down complex problems with many decision points
    - Planning with uncertainty where exploration is valuable
    - Problems with large branching factors that need guided search
    - Decision problems with trade-offs that require systematic exploration
    - Problems where some paths may lead to dead ends
    - Tasks that require balancing exploration vs. exploitation

    Parameters:
        thought: Your current thinking step (node in the search tree)
        thoughtNumber: Current thought number (minimum 1)
        totalThoughts: Estimated total thoughts needed (minimum 1)
        nextThoughtNeeded: Whether another thought step is needed
        nodeId: Unique identifier for this thought node (generated if not provided)
        parentId: ID of the parent thought node (if any)
        visits: Number of times this node has been visited (default 1)
        valueEstimate: Current estimated value of this state (0.0-1.0, default 0.5)
        childNodes: IDs of child nodes (automatically updated)
        depth: Depth in the tree (automatically calculated)
        action: Description of the action that led to this state
        explorationConstant: Controls exploration vs. exploitation (default 1.414)
        isRevision: Whether this revises previous thinking
        revisesThought: Which thought is being reconsidered
        branchFromThought: Branching point thought number
        branchId: Branch identifier
        needsMoreThoughts: If more thoughts are needed
    """
    input_data = {
        'thought': thought,
        'thoughtNumber': thoughtNumber,
        'totalThoughts': totalThoughts,
        'nextThoughtNeeded': nextThoughtNeeded,
        'nodeId': nodeId,
        'parentId': parentId,
        'visits': visits,
        'valueEstimate': valueEstimate,
        'childNodes': childNodes or [],
        'depth': depth,
        'action': action,
        'explorationConstant': explorationConstant,
        'isRevision': isRevision,
        'revisesThought': revisesThought,
        'branchFromThought': branchFromThought,
        'branchId': branchId,
        'needsMoreThoughts': needsMoreThoughts
    }
    
    return thinking_manager.process_thought(input_data)

if __name__ == "__main__":
    mcp.run()
