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
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import re
import json

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NodeType(Enum):
    ROOT = "root"
    THOUGHT = "thought"
    COMMAND = "command"
    SOLUTION = "solution"

class CommandStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"

@dataclass
class ActionState:
    """Represents the state of a command action."""
    command: str
    working_dir: str
    environment: Dict[str, str]
    dependencies: List[str]  # List of command IDs this action depends on
    prerequisites: List[str]  # List of conditions that must be met
    command_output: Optional[str] = None
    exit_code: Optional[int] = None
    execution_time: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    status: CommandStatus = CommandStatus.PENDING

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
        
        # Add node type and command-specific fields
        node_type = input_data.get('nodeType', NodeType.THOUGHT.value)
        
        # Handle action-specific state
        action_state = None
        if node_type == NodeType.COMMAND.value:
            # Create ActionState object
            action_state = ActionState(
                command=input_data.get('action', ''),
                working_dir=input_data.get('workingDir', ''),
                environment=input_data.get('environment', {}),
                dependencies=input_data.get('dependencies', []),
                prerequisites=input_data.get('prerequisites', []),
                command_output=input_data.get('commandOutput', ''),
                exit_code=input_data.get('exitCode'),
                execution_time=input_data.get('executionTime'),
                resource_usage=input_data.get('resourceUsage', {}),
                error_message=input_data.get('errorMessage', ''),
                status=CommandStatus(input_data.get('commandStatus', CommandStatus.PENDING.value))
            )
        
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
            'explorationConstant': exploration_constant,
            
            # Node type and command-specific fields
            'nodeType': node_type,
            'actionState': action_state
        }
    
    def calculate_ucb_score(self, node: Dict[str, Any], parent_visits: int) -> float:
        """
        Calculate UCB1 score for node selection.
        Now adjusts exploration based on failure context.
        
        Args:
            node (Dict[str, Any]): Node to calculate score for
            parent_visits (int): Number of visits to parent node
            
        Returns:
            float: UCB1 score for the node
        """
        # Base exploration constant
        exploration_constant = self.exploration_constant
        
        # Adjust exploration based on node context
        node_type = node.get('nodeType')
        if node_type == NodeType.COMMAND.value:
            result = node.get('result', {})
            if result.get('success') is False:
                error = result.get('error', '')
                
                # Increase exploration for failed nodes
                if 'undefined field' in error:
                    # Field errors suggest we're close - explore more
                    exploration_constant *= 1.5
                elif 'syntax error' in error:
                    # Syntax errors need more exploration of variations
                    exploration_constant *= 2.0
                elif 'timeout' in error:
                    # Timeouts might work with retry - moderate exploration
                    exploration_constant *= 1.3
                else:
                    # Other errors - significant exploration needed
                    exploration_constant *= 2.5
                    
        # Calculate exploitation term
        visits = node.get('visits', 0)
        if visits == 0:
            return float('inf')
            
        value = node.get('value', 0.0)
        exploitation = value / visits
        
        # Calculate exploration term
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / visits)
        
        return exploitation + exploration
    
    def get_recommended_nodes(self):
        """
        Get recommended nodes based on MCTS exploration.
        
        Returns:
            list: List of recommended nodes with their scores
        """
        recommendations = []
        
        # Get all nodes that haven't failed
        valid_nodes = []
        for node_id, node in self.node_map.items():
            # Skip nodes that have failed
            if node.get('classification') and node['classification'].status == CommandStatus.FAILURE:
                continue
            
            # Calculate UCB score
            visits = node.get('visits', 0)
            value = node.get('valueEstimate', 0.0)
            parent_visits = 0
            if node.get('parentId'):
                parent = self.node_map.get(node['parentId'])
                if parent:
                    parent_visits = parent.get('visits', 0)
                
            # UCB formula with exploration bonus
            if visits == 0:
                ucb_score = float('inf')  # Encourage exploring unvisited nodes
            else:
                exploitation = value
                exploration = math.sqrt(2 * math.log(parent_visits + 1) / visits)
                ucb_score = exploitation + exploration
            
            # Adjust score based on node type and context
            if node.get('nodeType') == NodeType.COMMAND.value:
                # If parent node failed, increase exploration
                if parent and parent.get('classification') and parent['classification'].status == CommandStatus.FAILURE:
                    ucb_score *= 1.5  # Encourage exploring different commands
                # If parent succeeded, slightly favor continuing sequence
                elif parent and parent.get('classification') and parent['classification'].status == CommandStatus.SUCCESS:
                    ucb_score *= 1.2  # Slightly favor continuing successful sequences
            
            valid_nodes.append({
                'nodeId': node_id,
                'score': ucb_score,
                'visits': visits,
                'value': value
            })
        
        # Sort by UCB score
        valid_nodes.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top recommendations
        return valid_nodes[:5]  # Return top 5 recommendations
    
    def update_node_value(self, node: Dict[str, Any], value: float) -> None:
        """
        Update node value and propagate up the tree.
        Now handles failure cases with partial credit based on error type.
        
        Args:
            node (Dict[str, Any]): Node to update
            value (float): New value to incorporate
        """
        # For action nodes, check result and adjust value
        if node.get('nodeType') == NodeType.COMMAND.value:
            result = node.get('result', {})
            if result.get('success') is False:
                error = result.get('error', '')
                
                # Check if this is a retry of a failed command
                parent = self.node_map.get(node.get('parentId'))
                if parent and parent.get('nodeType') == NodeType.COMMAND.value:
                    if parent.get('action') == node.get('action'):
                        # Retrying the same failed command gets a strong negative value
                        value = -1.0
                    else:
                        # Different command, assign partial credit based on error type
                        if 'undefined field' in error:
                            value = 0.3
                        elif 'syntax error' in error:
                            value = 0.2
                        elif 'timeout' in error:
                            value = 0.1
                        else:
                            value = 0.05
                else:
                    # First attempt at a command, assign partial credit
                    if 'undefined field' in error:
                        value = 0.3
                    elif 'syntax error' in error:
                        value = 0.2
                    elif 'timeout' in error:
                        value = 0.1
                    else:
                        value = 0.05
                    
        # Update node statistics
        node['visits'] = node.get('visits', 0) + 1
        node['value'] = node.get('value', 0.0) + value
        
        # Propagate up the tree
        parent_id = node.get('parentId')
        if parent_id:
            parent = self.node_map.get(parent_id)
            if parent:
                # For thought nodes, use max value
                if parent.get('nodeType') == NodeType.THOUGHT.value:
                    self.update_node_value(parent, max(value, parent.get('value', 0.0)))
                # For action nodes, use average
                else:
                    self.update_node_value(parent, value)
    
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
            prefix = '💭 Thought' if thought_data['nodeType'] == NodeType.THOUGHT.value else '⚡ Action'
            context = ''
        
        header = f"{prefix} {thought_data['thoughtNumber']}/{thought_data['totalThoughts']}{context}"
        
        node_id_short = thought_data['nodeId'][:8] + '...' if len(thought_data['nodeId']) > 8 else thought_data['nodeId']
        parent_id_short = (thought_data['parentId'][:8] + '...') if thought_data.get('parentId') and len(thought_data['parentId']) > 8 else 'root'
        
        mcts_info = f"Node: {node_id_short} | Parent: {parent_id_short} | Visits: {thought_data['visits']} | " \
                    f"Value: {thought_data['valueEstimate']:.3f} | Depth: {thought_data['depth']}"
        
        # Add action-specific info if this is an action node
        if thought_data['nodeType'] == NodeType.COMMAND.value and thought_data.get('actionState'):
            action_state = thought_data['actionState']
            mcts_info += f" | Command: {action_state.command} | Status: {action_state.status.value}"
            if action_state.error_message:
                mcts_info += f" | Error: {action_state.error_message[:50]}..."
        
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
    
    def extract_commands_from_thought(self, thought: str) -> List[str]:
        """Extract commands from a thought's content."""
        # Look for <action> tags
        action_pattern = r'<action>(.*?)</action>'
        action_matches = re.findall(action_pattern, thought, re.DOTALL)
        
        if not action_matches:
            return []
        
        # Parse the JSON array inside the action tags
        try:
            actions = json.loads(action_matches[0])
            return [action['action_input'] for action in actions if 'action_input' in action]
        except json.JSONDecodeError:
            return []

    def process_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a thought and return the response."""
        try:
            validated_input = self.validate_thought_data(input_data)
            
            # Extract commands if this is a thought node
            if validated_input['nodeType'] == NodeType.THOUGHT.value:
                commands = self.extract_commands_from_thought(validated_input['thought'])
                if commands:
                    # Create the first action node as child of thought
                    first_action = {
                        'nodeId': self.generate_node_id(),
                        'parentId': validated_input['nodeId'],
                        'nodeType': NodeType.COMMAND.value,
                        'action': commands[0],
                        'status': CommandStatus.PENDING.value,
                        'depth': validated_input['depth'] + 1
                    }
                    self.node_map[first_action['nodeId']] = first_action
                    validated_input['childNodes'].append(first_action['nodeId'])
                    
                    # Create subsequent action nodes, each as child of previous action
                    current_parent = first_action
                    for command in commands[1:]:
                        next_action = {
                            'nodeId': self.generate_node_id(),
                            'parentId': current_parent['nodeId'],
                            'nodeType': NodeType.COMMAND.value,
                            'action': command,
                            'status': CommandStatus.PENDING.value,
                            'depth': current_parent['depth'] + 1
                        }
                        self.node_map[next_action['nodeId']] = next_action
                        current_parent['childNodes'] = [next_action['nodeId']]
                        current_parent = next_action
            
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
            recommendations = self.get_recommended_nodes()
            
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

    def select_node(self, node_id: str) -> str:
        """
        Select the next node to explore using UCB1.
        Now handles transitions between thought and action nodes.
        
        Args:
            node_id (str): ID of the current node
            
        Returns:
            str: ID of the selected child node
        """
        node = self.node_map.get(node_id)
        if not node:
            return None
            
        # Get all child nodes
        children = [
            child for child in self.node_map.values()
            if child.get('parentId') == node_id
        ]
        
        if not children:
            return None
            
        # Calculate UCB scores for all children
        parent_visits = node.get('visits', 0)
        scored_children = [
            {
                'node': child,
                'ucbScore': self.calculate_ucb_score(child, parent_visits)
            }
            for child in children
        ]
        
        # Sort by UCB score
        scored_children.sort(key=lambda x: x['ucbScore'], reverse=True)
        
        # Consider node types in selection
        current_type = node.get('nodeType')
        
        if current_type == NodeType.THOUGHT.value:
            # For thought nodes, prefer exploring new actions
            action_children = [
                child for child in scored_children
                if child['node'].get('nodeType') == NodeType.COMMAND.value
            ]
            
            if action_children:
                # Prioritize unvisited actions
                unvisited_actions = [
                    action for action in action_children
                    if action['node'].get('visits', 0) == 0
                ]
                if unvisited_actions:
                    return unvisited_actions[0]['node']['id']
                    
                # Otherwise use UCB for actions
                return action_children[0]['node']['id']
                
        elif current_type == NodeType.COMMAND.value:
            # For action nodes, prefer exploring new thoughts
            thought_children = [
                child for child in scored_children
                if child['node'].get('nodeType') == NodeType.THOUGHT.value
            ]
            
            if thought_children:
                # Prioritize unvisited thoughts
                unvisited_thoughts = [
                    thought for thought in thought_children
                    if thought['node'].get('visits', 0) == 0
                ]
                if unvisited_thoughts:
                    return unvisited_thoughts[0]['node']['id']
                    
                # Otherwise use UCB for thoughts
                return thought_children[0]['node']['id']
                
        # Default to highest UCB score if no type-specific selection
        return scored_children[0]['node']['id']

    def expand_node(self, node_id: str) -> str:
        """
        Expand the current node by adding a new child node.
        Now handles expansion after failures by considering alternative approaches.
        
        Args:
            node_id (str): ID of the current node
            
        Returns:
            str: ID of the newly created child node
        """
        node = self.node_map.get(node_id)
        if not node:
            return None
            
        # Get node type and context
        node_type = node.get('nodeType')
        parent_thought = None
        
        # If expanding from an action node, get the parent thought
        if node_type == NodeType.COMMAND.value:
            parent_thought = self.node_map.get(node.get('parentId'))
            
        # Create new node based on type
        if node_type == NodeType.THOUGHT.value:
            # After a thought, try an action
            new_node = {
                'id': f'node_{int(time.time()*1000)}_{random.randint(1000, 9999)}',
                'nodeType': NodeType.COMMAND.value,
                'parentId': node_id,
                'visits': 0,
                'value': 0.0,
                'thought': node.get('thought', ''),
                'action': '',  # Will be filled by the action generation
                'result': None
            }
            
        elif node_type == NodeType.COMMAND.value:
            # After an action, create a new thought
            # If the action failed, consider alternative approaches
            if node.get('result', {}).get('success') is False:
                # Get failure context
                error = node.get('result', {}).get('error', '')
                action = node.get('action', '')
                
                # Create thought about alternative approach
                new_thought = f"Previous action failed with error: {error}. Considering alternative approach..."
                
            else:
                # Normal expansion after successful action
                new_thought = "Evaluating the result and considering next steps..."
                
            new_node = {
                'id': f'node_{int(time.time()*1000)}_{random.randint(1000, 9999)}',
                'nodeType': NodeType.THOUGHT.value,
                'parentId': node_id,
                'visits': 0,
                'value': 0.0,
                'thought': new_thought,
                'action': None,
                'result': None
            }
            
        # Add the new node to the tree
        self.node_map[new_node['id']] = new_node
        return new_node['id']

    def split_thoughts_from_text(self, think_text: str) -> List[str]:
        """
        Split a string of concatenated thoughts (from <think> tags) into a list of individual thoughts.
        Splits on double newlines.
        Args:
            think_text (str): The text containing all thoughts
        Returns:
            List[str]: List of individual thought strings
        """
        if not think_text.strip():
            return []
        # Split on two or more newlines
        thoughts = [t.strip() for t in re.split(r'\n\s*\n', think_text) if t.strip()]
        return thoughts

    def organize_thoughts_into_chain(self, thoughts: List[str], total_thoughts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Organize a list of thought strings into a chain of thought nodes.
        Each node is created using process_thought, and linked as the child of the previous node.
        Args:
            thoughts (List[str]): List of thought strings
            total_thoughts (Optional[int]): Total number of thoughts (for node metadata)
        Returns:
            List[Dict[str, Any]]: List of node dicts in chain order
        """
        if not thoughts:
            return []
        if total_thoughts is None:
            total_thoughts = len(thoughts)
        nodes = []
        parent_id = None
        for idx, thought in enumerate(thoughts):
            node_data = {
                'thought': thought,
                'thoughtNumber': idx + 1,
                'totalThoughts': total_thoughts,
                'nextThoughtNeeded': (idx < len(thoughts) - 1),
                'nodeType': NodeType.THOUGHT.value,
                'parentId': parent_id
            }
            node_result = self.process_thought(node_data)
            node_id = node_result.get('currentNodeId')
            parent_id = node_id
            nodes.append(self.node_map[node_id])
        return nodes

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
