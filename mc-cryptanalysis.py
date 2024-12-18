#!/usr/bin/env python3
import argparse
import json
import random
import string
import math
from collections import defaultdict

class MarkovCryptanalysis:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_transitions = 0

    def clean_text(self, text):
        """Convert text to lowercase and remove symbols."""
        cleaned = []
        for c in text:
            if c.isalpha():
                cleaned.append(c.lower())
            elif c.isspace() and cleaned and cleaned[-1] != ' ':
                cleaned.append(' ')

        return ''.join(cleaned)
        
    # def clean_text(self, text):
    #     """Old cleaning method used since I had a weird corpus."""
    #     cleaned = []
    #     for i, c in enumerate(text):
    #         if c.isalpha():
    #             cleaned.append(c.lower())
    #         elif c.isspace():
    #             if cleaned and cleaned[-1] != ' ':
    #                 cleaned.append(' ')
    #         else:  # it's a symbol
    #             # If symbol is between letters, replace with space (this was due to a previous data set i used)
    #             if (i > 0 and i < len(text) - 1 and 
    #                 text[i-1].isalpha() and text[i+1].isalpha()):
    #                 if cleaned and cleaned[-1] != ' ':
    #                     cleaned.append(' ')
    #             # If symbol follows a space, remove that space (this was due to a previous data set i used)
    #             elif cleaned and cleaned[-1] == ' ':
    #                 cleaned.pop()
        
    #     return ''.join(cleaned).strip()
    
    def train(self, text):
        """Train the Markov model on a given text."""
        text = self.clean_text(text)
        
        # Process transitions within words
        words = text.split()
        for word in words:
            for i in range(len(word) - 1):
                current = word[i]
                next_char = word[i + 1]
                self.transitions[current][next_char] += 1
                self.total_transitions += 1
        
        # Convert transition counts to probabilities
        for char in self.transitions:
            total = sum(self.transitions[char].values())
            for next_char in self.transitions[char]:
                self.transitions[char][next_char] /= total
    
    def transition_score(self, text):
        """Score the text based on transition probabilities."""
        score = 0
        words = text.split()
        
        for word in words:
            for i in range(len(word) - 1):
                current = word[i]
                next_char = word[i + 1]
                if self.transitions[current][next_char] > 0:
                    score += math.log(self.transitions[current][next_char])
        
        return score
    
    def hill_climb_single(self, ciphertext, current_key, iterations, temperature=1.0):
        """Single hill climbing attempt."""
        alphabet = list(string.ascii_lowercase)
        best_key = current_key.copy()
        best_score = float('-inf')
        best_decrypted = ''
        current_score = float('-inf')
        
        cooling_rate = 0.99998
        min_temperature = 0.01
        
        stagnant_iterations = 0
        max_stagnant = 5000  # Reset if stuck for too long
        
        for _ in range(iterations):
            if stagnant_iterations > max_stagnant:
                temperature = 1.0
                stagnant_iterations = 0
            
            # Swap 2 letters in key
            a, b = random.sample(alphabet, 2)
            new_key = current_key.copy()
            new_key[a], new_key[b] = new_key[b], new_key[a]
            
            # Decrypt with key and score
            reverse_key = {v: k for k, v in new_key.items()}
            decrypted = ''.join(reverse_key.get(c, c) if c.isalpha() else c for c in ciphertext)
            score = self.transition_score(decrypted)
            
            # Update best score if improved
            if score > best_score:
                best_score = score
                best_key = new_key.copy()
                best_decrypted = decrypted
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            
            # Accept if score is better or based on temperature to not be stuck
            if score > current_score or random.random() < math.exp((score - current_score) / temperature):
                current_key = new_key
                current_score = score
            
            temperature = max(temperature * cooling_rate, min_temperature)
        
        return best_decrypted, best_score, best_key
    
    def hill_climb(self, ciphertext, iterations=50000):
        """Multiple hill climbing attempts with different starting points."""
        ciphertext = self.clean_text(ciphertext)
        alphabet = list(string.ascii_lowercase)
        best_overall_score = float('-inf')
        best_overall_decrypted = ''
        
        num_attempts = 10
        iterations_per_attempt = iterations // num_attempts
        
        for _ in range(num_attempts):
            # Start with random key
            current_key = dict(zip(alphabet, random.sample(alphabet, len(alphabet))))
            
            # Multiple climbs from this starting point with different temperatures
            for temp in [1.0, 0.5, 0.1]:
                decrypted, score, key = self.hill_climb_single(
                    ciphertext, 
                    current_key,
                    iterations_per_attempt // 3,
                    temperature=temp
                )
                
                if score > best_overall_score:
                    best_overall_score = score
                    print(f"Best score so far: {best_overall_score}")
                    best_overall_decrypted = decrypted
                    current_key = key
        
        return best_overall_decrypted
    
    def encrypt(self, text):
        """Encrypt text using a random substitution cipher."""
        text = self.clean_text(text)
        
        # Generate random substitution key
        alphabet = list(string.ascii_lowercase)
        shuffled = list(string.ascii_lowercase)
        random.shuffle(shuffled)
        key = dict(zip(alphabet, shuffled))
        
        encrypted = ''.join(key.get(c, c) if c.isalpha() else c for c in text)
        return text, encrypted, key
    
    def save_model(self, filepath):
        """Save the trained model to a JSON file."""
        model_data = {
            'transitions': {k: dict(v) for k, v in self.transitions.items()},
            'total_transitions': self.total_transitions
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath):
        """Load a trained model from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.transitions = defaultdict(lambda: defaultdict(float))
        for k, v in model_data['transitions'].items():
            self.transitions[k] = defaultdict(float, v)
        
        self.total_transitions = model_data['total_transitions']

def format_mapping(key):
    """Format the encryption/decryption mapping in a readable way."""
    return "\n".join([f"{k} -> {v}" for k, v in sorted(key.items())])

def read_input(input_source):
    """Read input from either a file or direct text."""
    if input_source.endswith('.txt'):
        with open(input_source, 'r', encoding='utf-8') as f:
            return f.read()
    return input_source

def main():
    parser = argparse.ArgumentParser(description='Markov Chain Cryptanalysis Tool')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train on input text')
    train_parser.add_argument('input', type=str, 
                            help='Input text file path or direct text')
    train_parser.add_argument('model_file', type=str, 
                            help='Output model file path')
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a message')
    encrypt_parser.add_argument('input', type=str, 
                              help='Input text file path or direct text')
    encrypt_parser.add_argument('output_file', type=str, 
                              help='Output file for ciphertext')
    
    # Decrypt command
    decrypt_parser = subparsers.add_parser('decrypt', help='Decrypt a message')
    decrypt_parser.add_argument('input', type=str, 
                              help='Input text file path or direct text')
    decrypt_parser.add_argument('model_file', type=str, 
                              help='Trained model file')
    decrypt_parser.add_argument('--iterations', type=int, default=50000,
                              help='Number of iterations for hill climbing')
    
    args = parser.parse_args()
    
    analyzer = MarkovCryptanalysis()
    
    if args.command == 'train':
        text = read_input(args.input)
        analyzer.train(text)
        analyzer.save_model(args.model_file)
        print(f"Model trained and saved to {args.model_file}")
    
    elif args.command == 'encrypt':
        text = read_input(args.input)
        plaintext, ciphertext, key = analyzer.encrypt(text)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(ciphertext)
        
        print(f"Plaintext:  {plaintext}")
        print(f"\nCiphertext: {ciphertext}")
        print(f"\nSubstitution Mapping:")
        print(format_mapping(key))
        print(f"\nCiphertext saved to {args.output_file}")
    
    elif args.command == 'decrypt':
        analyzer.load_model(args.model_file)
        text = read_input(args.input)
        decrypted = analyzer.hill_climb(text, args.iterations)
        print(f"Decrypted: {decrypted}")
        print(f"\nOriginal:  {text}")

if __name__ == '__main__':
    main()