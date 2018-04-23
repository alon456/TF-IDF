#### Globals
pref		= "C:/spark/spark/bin/results/"
const_e 	= 2.71828 #the constant value of e, for calculating lan value
_write_temp = 200 #temp for writing '-' into output file for seperation

#### Imports
from math import *
from pyspark import SparkContext , SparkConf
conf = SparkConf().setAppName ('MyFirstStandaloneApp')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

class TF_IDF():
	#class attributes
	num_of_files	= 0
	documents		= []
	res_tf_lst		= [] #result of TF function
	res_idf 		= []
	res_tfidf 		= []

	#class init function
	def __init__(self, num_of_files, documents):
		self.num_of_files 	= num_of_files
		self.documents 		= documents

	#the function that calculate TD, IDF and TD.IDF
	def calc_tf_idf(self):
		#calculate TF :
		for i in range(self.num_of_files):
			file_RDD = sc.textFile(pref + self.documents[i])
			self.func_TF(file_RDD, i)
		
		#calculate IDF :
		self.func_IDF()
		
		#Calculate TF.IDF :
		self.func_calc_tfidf()

	#TF function
	def func_TF(self, file_RDD, i):
		words =  file_RDD.flatMap(lambda x : x.split(" ")) #get words from document
		pairs = words.map(lambda word : (word.lower() , 1)) #pair each word with the number 1
		
		#Total words in document
		num_words = words.count() #how much words in this document
		
		if (num_words == 0): #0 words -> empty file -> return [] (to prevent division by 0)
			res_tf = sc.parallelize([])
			self.res_tf_lst.append(res_tf)
			return #done
		
		#Number of times word t appears in document
		wordCount = pairs.reduceByKey( lambda x , y : x + y ) #for each word in document count it's appears
		
		res_tf = wordCount.map(lambda (word, val) : (word, float(val) / num_words)) #for each word calculate it's TF value
		
		self.res_tf_lst.append(res_tf)

	#Merge all the words in corpus into one list	
	def merge_all_words(self):
		all_words = self.res_tf_lst[0].keys() #init with the words from the first document
		for i in range(1, self.num_of_files): #union the words from all of the documents
			all_words = all_words.union(self.res_tf_lst[i].keys()) 
		
		#all_words will contain the words from all of the corpus. 
		#if a word W appears in k different documents W will appear k times in all_words.
		
		return all_words #return result
		
	#IDF function
	def func_IDF(self):
		all_words = self.merge_all_words() #merge all words in corpus
		
		if (all_words.count() == 0): #0 words -> empty corpus -> return [] (to prevent division by 0)
			self.res_idf = sc.parallelize([])
			return #done
		
		paired_all_words = all_words.map(lambda word : (word, 1)) #pair each word in corpus with the number 1
		words_appear_num = paired_all_words.reduceByKey(lambda x, y : x + y) #for each word in corpus count it's appears
			
		f_num = float(self.num_of_files) #number of files in corpus as float
		self.res_idf = words_appear_num.mapValues(lambda x : log(f_num / x, const_e)) #for each word calculate it's IDF value

	#Calculate the TF.IDF	
	def func_calc_tfidf(self):
		for i in range(self.num_of_files):
			#join by the key (word) the tf and idf values
			joined = self.res_tf_lst[i].join(self.res_idf)
			#calculate for the words in i'th document their tf.idf value
			tf_idf = joined.mapValues(lambda (idf_val, tf_val) : tf_val * idf_val)
			self.res_tfidf.append(tf_idf)
		
	#Write final TF.IDF results to output file
	def func_write_results(self):
		output_file = open(pref + "results.txt", 'w')
		for i in range(self.num_of_files):
			#write the td.idf score of words from the i'th document
			output_file.write("\n\n" + str('-' * _write_temp)+"\n\n")
			output_file.write(str(i) + "'th document, TF.IDF results :\n")
			temp = self.res_tfidf[i].sortBy(lambda a: -a[1]) #sort by value descending
			output_file.write(str(temp.collect()))
			output_file.write("\n\n" + str('-' * _write_temp)+"\n\n")
		output_file.close()

	#####END OF CLASS####

def main():	
	#adjust arguments
	num_of_files = 5 #number of files
	documents = ["f1.txt", "f2.txt", "f3.txt", "f4.txt","f5.txt"] #files names

	my_tf_idf = TF_IDF(num_of_files, documents) #init class with related arguments

	my_tf_idf.calc_tf_idf() #calculate tf.idf result
	
	my_tf_idf.func_write_results()#Write final TF.IDF results to output file
	
	#####END OF MAIN####

if __name__ == "__main__":
    main()
