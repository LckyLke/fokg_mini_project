
import re
class KG:
	def __init__(self, kb_path):
		self.kb_path = kb_path
		self.triples = self.load_kb(kb_path)
		self.entities = self.get_entities(self.triples)
		self.relations = self.get_relations(self.triples)
		self.num_entities = len(self.entities)
		self.num_relations = len(self.relations)

		self.entity_idx = {self.entities[i]: i for i in range(len(self.entities))}
		self.relation_idx = {self.relations[i]: i for i in range(len(self.relations))}
		self.triples_idx = [(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]) for s, p, o in self.triples]
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


	
