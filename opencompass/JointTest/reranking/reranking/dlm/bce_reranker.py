from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'who proposed the geocentric theory'
passages = [
    'For Earth-centered it was  Geocentric Theory proposed by greeks under the guidance of Ptolemy and Sun-centered was Heliocentric theory proposed by Nicolas Copernicus in 16th century A.D. In short, Your Answers are: 1st blank - Geo-Centric Theory. 2nd blank - Heliocentric Theory.',
    'Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.he geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.',
    'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.opernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.',
    'Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.',
    'Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect.opernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.',
    'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.imple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect. You might want to check out one article on the history of the geocentric model and one regarding the geocentric theory.',
    'Nicolaus Copernicus (b. 1473âd. 1543) was the first modern author to propose a heliocentric theory of the universe. From the time that Ptolemy of Alexandria (c. 150 CE) constructed a mathematically competent version of geocentric astronomy to Copernicusâs mature heliocentric version (1543), experts knew that the Ptolemaic system diverged from the geocentric concentric-sphere conception of Aristotle.',
    'A Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth. Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth.',
    'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.',
    "After 1,400 years, Copernicus was the first to propose a theory which differed from Ptolemy's geocentric system, according to which the earth is at rest in the center with the rest of the planets revolving around it."
]

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# init reranker model
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)
print(scores)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)
print(rerank_results)