
import re
class KG:
	def __init__(self, kb_path=None, test_path=None, train_path=None):
		self.kb_path = kb_path
		if kb_path is not None:
			self.triples = self.load_kb(kb_path)
		elif test_path is not None:
			self.triples = self.load_test_data_to_spo(test_path)
		elif train_path is not None:
			self.triples, self.truth_str = self.load_test_data_with_truth(train_path)
			

		self.entities = self.get_entities(self.triples)
		self.relations = self.get_relations(self.triples)
		self.num_entities = len(self.entities)
		self.num_relations = len(self.relations)

		self.entity_idx = {self.entities[i]: i for i in range(len(self.entities))}
		self.relation_idx = {self.relations[i]: i for i in range(len(self.relations))}
		self.triples_idx = [(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]) for s, p, o in self.triples]

		if train_path is not None:
			# Initialize `self.truth` with ID-based keys
			self.truth = {
				(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]): self.truth_str[(s, p, o)]
				for s, p, o in self.triples
				if (s, p, o) in self.truth_str
			}

		print('KG loaded with {} triples, {} entities, and {} relations'.format(len(self.triples), len(self.entities), len(self.relations)))
	@staticmethod
	def load_kb(kb_path):
		triples = []
		with open(kb_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				
				# Each line is expected to end with a '.'
				# A typical line looks like:
				# <http://rdf.freebase.com/ns/m.06mzp> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://rdf.freebase.com/ns/organization.organization_founder> .
				#
				# or
				#
				# <http://example.org/person> <http://example.org/hasName> "John Doe" .
				#
				# We can first remove the trailing '.' and then split into 3 parts: h, r, t.
				# We'll split only twice, so quoted strings in t are preserved.
				
				if line.endswith('.'):
					line = line[:-1].strip()  # Remove trailing period and surrounding spaces
				
				# Now split into h, r, and t
				# Use a maximum of two splits so that the tail (t) isn't fragmented
				parts = line.split(' ', 2)
				if len(parts) != 3:
					# If the line doesn't properly split into three parts -> skip 
					continue
				
				h, r, t = parts
				
				# Now we have h, r, t. For example:
				# h = <http://rdf.freebase.com/ns/m.06mzp>
				# r = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
				# t = <http://rdf.freebase.com/ns/organization.organization_founder> 
				# or t = "this is a string"
				
				# The head and relation are URIs enclosed in <>, and the tail can be a URI or a quoted literal.
				# No further parsing may be needed unless you want to strip the angle brackets or quotes.
				
				triples.append((h, r, t))
		
		return triples


	@staticmethod
	def get_entities(triples):
		entities = set()
		for h, _, t in triples:
			entities.add(h)
			entities.add(t)
		return list(entities)
	
	@staticmethod
	def get_relations(triples):
		relations = set()
		for _, r, _ in triples:
			relations.add(r)
		return list(relations)

	@staticmethod
	def load_test_data_to_spo(test_path):
		"""
		Convert test data in expanded RDF format to standard (subject, predicate, object) triples.

		Args:
			test_path (str): Path to the test data file.

		Returns:
			List[Tuple[str, str, str]]: List of triples in (subject, predicate, object) format.
		"""
		triples_dict = {}
		with open(test_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line or not line.endswith('.'):
					continue
				
				# Remove the trailing period and split into parts
				line = line[:-1].strip()
				parts = line.split(' ', 2)
				if len(parts) != 3:
					continue
				
				subj, pred, obj = parts
				
				# Group lines by the subject identifier
				if subj not in triples_dict:
					triples_dict[subj] = {}
				triples_dict[subj][pred] = obj
		
		# Create (subject, predicate, object) triples
		spo_triples = []
		for _, properties in triples_dict.items():
			# Check if required properties are available
			if (
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>' in properties and
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>' in properties and
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>' in properties
			):
				s = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>']
				p = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>']
				o = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>']
				spo_triples.append((s, p, o))
		print('Loaded {} test triples'.format(len(spo_triples)))
		print(spo_triples[0])
		return spo_triples

	@staticmethod
	def load_test_data_with_truth(test_path):
		"""
		Convert test data with truth values into standard (subject, predicate, object) triples
		and store their truth values.

		Args:
			test_path (str): Path to the test data file.

		Returns:
			List[Tuple[str, str, str]]: List of triples in (subject, predicate, object) format.
			Dict[Tuple[str, str, str], float]: Mapping of triples to truth values.
		"""
		triples_dict = {}
		truth_values = {}

		with open(test_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line or not line.endswith('.'):
					continue

				# Remove the trailing period and split into parts
				line = line[:-1].strip()
				parts = line.split(' ', 2)
				if len(parts) != 3:
					continue

				subj, pred, obj = parts

				# Group lines by the subject identifier
				if subj not in triples_dict:
					triples_dict[subj] = {}
				triples_dict[subj][pred] = obj

		# Create (subject, predicate, object) triples and truth values
		spo_triples = []
		for subj, properties in triples_dict.items():
			if (
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>' in properties and
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>' in properties and
				'<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>' in properties
			):
				s = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>']
				p = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>']
				o = properties['<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>']
				spo_triples.append((s, p, o))

				# Parse truth value if available
				if '<http://swc2017.aksw.org/hasTruthValue>' in properties:
					truth_str = properties['<http://swc2017.aksw.org/hasTruthValue>']
					try:
						# Extract only the numeric part before ^^
						truth_value = float(truth_str.split('^^')[0].strip('"'))
						truth_values[(s, p, o)] = truth_value
					except ValueError:
						print('Failed to parse truth value:', truth_str)
						truth_values[(s, p, o)] = 1.0  # Default truth value if parsing fails

		return spo_triples, truth_values



