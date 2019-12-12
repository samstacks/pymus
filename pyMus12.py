__author__ = 'Samuel Stathakos'

#imports
import wave
import struct
import math
import numpy as np
from functools import partial
import random
from fractions import Fraction
import itertools as it

print ('hello world')

#global variables
filename = "out.wav" # + parametrize...
framerate = 44100.0
base_freq = 16.0018912505 # middle A = 432 Hz

class Note:

	NOTES = [range(12)] #0-12 note halfsteps

	def __init__(self, note, octave=4):
		if note == None:
			self.index = None # rest note
			return
		assert isinstance(note,int)
		self.octave = octave % 8
		self.index = note % 12

	#transpose note by so many halfsteps
	def transpose(self, halfsteps=1):
		if self.index == None:
			return
		octave_delta, self.index = divmod(self.index + halfsteps, 12)
		self.octave = (self.octave + octave_delta) % 8 
		return

	def frequency(self):
		note_frequency = base_freq * 2.0 ** (float(self.index) / 12.0)
		return note_frequency * (2.0 ** self.octave)

class Tone:
	# frequency and amplitude
	def __init__(self, pitch, amp=0.8):
		assert isinstance(pitch, float) or isinstance(pitch, Note)
		if isinstance(pitch, float):
			self.frequency = pitch
		elif isinstance(pitch, Note):
			self.frequency = pitch.frequency()
		self.amplitude = amp

	def freqMod(self,op,mod):
		self.frequency = op(self.frequency, mod)

	def ampMod(self,op,mod):
		self.amplitude = op(self.amplitude, mod)
        
	def source():
		# checks
		if self.frequency <= base_freq:
			self.frequency = base_freq
		assert framerate >= self.frequency * 2 # Nyquist rate
		assert framerate != 0.0
		assert period != 0
		assert 0.0 <= self.amplitude <= 1.0 # and " " 
		assert self.frequency >= 0.0
        
		# compute sine wave form of the tone
		period = int(framerate / self.frequency)
		lookup_table = [float(self.amplitude) * np.sin(2.0*np.pi*float(self.frequency)*(float(i%period)/float(framerate))) for i in xrange(period)]
		return (lookup_table[i%period] for i in count(0))

class Sound:
	#note, harmonic, amp, dur, chan

    def __init__(self, note, amp=0.8, harmonic=[], harmAmp=[], dur=1000, chan='C'):
        self.note = note #check is Note
        self.amplitude = amp #check between 0-1 
        self.harmonic = harmonic #check list of Fraction
        self.harmonicAmp = harmAmp
        self.duration = dur #check int 
        self.channel = chan #check L or R
        
    def tone_array(self):
    	normalizeAmp = (len(self.harmonic)+1.0)
    	note_tone = Tone(self.note, self.amplitude / normalizeAmp)
    	tones = np.empty(len(self.harmonic)+1, object)
    	tones[0] = note_tone
    	for i, h in enumerate(self.harmonic):
    		assert type(h) == Fraction
    		new_tone = Tone(note_tone.frequency*h.numerator/h.denominator, self.harmonicAmp[i]/normalizeAmp)
    		tones[i+1] = new_tone 
    	return tones
    		
class SoundSeq:

	# takes np array of Sound
    def __init__(self, sounds):
        assert isinstance(sounds, np.ndarray)
        self.seq = sounds # sequence

    def repeat(self, repeats=2):
    	assert repeats > 1 and type(repeats) == int
    	self.seq += np.tile(self.seq, repeats - 1)
    	return

    def roll(self, shift=1):
    	assert type(shift) == int
    	self.seq = np.roll(self.seq, shift)
    	return

    def flip(self):
    	self.seq = self.seq[::-1]
    	return

    #transpose
    def transpose(self, halfsteps):
    	for sound in self.seq:
    		octave_delta, sound.note.index = divmod(sound.note.index + halfsteps, 53)
    		sound.note.octave = (sound.note.octave + octave_delta) % 8
    	return
        
    #copy methods
    def slice_seq(self, arr):
    	return SoundSeq(self.seq[arr])
    	
    def section(self, init, end, jump=1):
    	return SoundSeq(self.seq[init:end:jump])
    
    def add(self, seq2):
    	return SoundSeq(np.concatenate((self.seq, seq2.seq)))

class SoundSeqToWav:

	sampwidth = 2 #in bytes
	max_amp = float(int((2 ** (sampwidth * 8)) / 2) - 1)
	nchannels = 2
	comptype = "NONE"
	compname = "not compressed"
    
	def __init__(self, soundseq_l, soundseq_r):
		assert isinstance(soundseq_l, SoundSeq) and isinstance(soundseq_r, SoundSeq)
		assert len(soundseq_r.seq) == len(soundseq_l.seq)
		
		nsamples = 0 # number of samples
		self.nframes = 0 # number of frames
		
		n = [] # to hold samples of Source of SoundSeq
		
		for i, sound in enumerate(soundseq_l.seq):
			assert isinstance(sound, Sound)
			tones = sound.tone_array()
			s = np.empty(len(tones)+1, object) #tone sources
			for j, tone in enumerate(tones):
				s[j] = tone.source
			nsamples = framerate / 1000.0 * sound.duration
			self.nframes += nsamples
			channel_l = tuple(s) # nl	
			
			sound_r = soundseq_r.seq[i] # same index of left sequence
			assert isinstance(sound_r, Sound)
			tones2 = sound_r.tone_array()
			s2 = np.empty(len(tones2)+1, object)
			for k, tone in enumerate(tones2):
				s2[k] = tone.source
			assert sound_r.duration == sound.duration # same nsamples
			self.nframes += nsamples
			channel_r = tuple(s2)
			
			channels = (channel_l, channel_r)
			n.append(self.compute_samples(channels,nsamples))

		outsamp = it.chain.from_iterable(n)
		self.write_wavefile(outsamp)

	def compute_samples(self, channels, nsamples=None):
		return it.islice(zip(list(map(sum, zip(channel)) for channel in channels)), int(nsamples))

	def grouper(self, n, iterable, fillvalue=None):
		"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
		args = [iter(iterable)] * n
		return  it.zip_longest(fillvalue=None, *args)

	def write_wavefile(self, samples, bufsize=2048*4):
		"Write samples to a wavefile"
		if self.nframes is None:
			self.nframes = -1

		w = wave.open(filename, 'w')
		w.setparams((self.nchannels,self.sampwidth,framerate,self.nframes,self.comptype,self.compname))

      	#split samples into chunks (to reduce memory consumption, improve performance)
		for chunk in self.grouper(bufsize, samples):
			frames = ''.join(''.join(struct.pack('h', np.int16(self.max_amp * sample)) for sample in channels) for channels in chunk if channels is not None)
			w.writeframesraw(frames)

		w.close()

def scaleFree(num_node):
	# preferential attachment algorithm for scale free network
	
	node = [[1], [0,2], [1]]
	deg = [1,2,1]
	desire_deg = math.floor(math.pow(num_node,2) / 2.0 * 0.6180339)
	for i in range(3, num_node):
		connections = []
		for j in range(i):
			to_des = float(desire_deg - sum(deg)) / float(num_node - i)
			p = (float(deg[j]) / sum(deg)) * to_des
			if(random.random() < p):
				connections.append(j)
				node[j].append(i)
				deg[j] += 1
		deg.append(len(connections))
		node.append(connections)
	return node

#makeSeq
def makeSeq(network, depth=5):
	#in and out a list
	#find most connected node
	assert type(network) == list
	#m_node = network[max(i) for i,len(node) in enumerate(network))]	

	m_node = np.argmax([len(node) for node in network])
	#random sample from most connected
	#random sample from random sample... a sample from connected component
	seq = [m_node] # some nodes sampled from network
	s = len(seq)
	
	
	for i in range(depth): # range of...how much sequence?
		newSeq = []
		for node in seq:
			newSeq += network[node]
			
		seq = newSeq
		
	return seq

def makeSoundSeq(seq, l_r_dur):
	# seq, l_r_dur -> 2 SoundSeq (l and r)
	
	lSeq = seq[::2]
	print ('lseq and seq[::2]: ', lSeq, seq[::2])
	rSeq = seq[1::2]
	if (len(lSeq)>len(rSeq)):
		rSeq += [lSeq[-1]]
	assert(len(lSeq)==len(rSeq))
	
	lSoundSeq = np.empty(len(lSeq), object)
	#lSeqToSound = l_r_dur[0] #[(ind,oct,amp,harm,hAmp),...]
	
	for i, node in enumerate(lSeq):
		#create sound
		#for #each of durations, leng of l r dur
		for entry in l_r_dur:
			print ('lseqtosound = entry 0 :', entry[0])
			lSeqToSound = entry[0]
			print ('dur = entry 2 :', entry[2])
			dur = entry[2]
			s = lSeqToSound[node%len(lSeqToSound)]
			print ('s: ', s)
			if s == None:
				lSoundSeq[i] = Sound(Note(0,0),0.0,[],[],dur,'L')
			else:
				ind, octa, amp, harm, hAmp = s
				lSoundSeq[i] = Sound(Note(ind,octa),amp,[harm],[hAmp],dur,'L')
				
	rSoundSeq = np.empty(len(lSeq), object)
	#lSeqToSound = l_r_dur[0] #[(ind,oct,amp,harm,hAmp),...]
	
	for i, node in enumerate(rSeq):
		#create sound
		#for #each of durations, leng of l r dur
		for entry in l_r_dur:
			rSeqToSound = entry[1]
			dur = entry[2]
			s = rSeqToSound[node%len(rSeqToSound)]
			if s == None:
				rSoundSeq[i] = Sound(Note(0,0),0.0,[],[],dur,'R')
			else:
				ind, octa, amp, harm, hAmp = s
				rSoundSeq[i] = Sound(Note(ind,octa),amp,[harm],[hAmp],dur,'R')
				
	return (SoundSeq(lSoundSeq),SoundSeq(rSoundSeq))
	
def seqToSound(partition_ind,octaves,durations,amplitudes,harmonic):
	#returns num_node, l_r_dur from indices, octs, durs, amps, h, partition, 
	num_node = 0
	lSeqToSound = []
	rSeqToSound = []
	l_r_durToSound = []

	# partition_ind is like [[10,3,5,4], [6,7,14,12], ...]
	for dur in random.sample(durations,1):
		print(partition_ind, "partition ind :")
		for inds in random.sample(partition_ind, 1):
			#inds is like [10,3,5,4]
			print(inds, "inds :")
			for ind in inds:
				if ind == 12:
					lSeqToSound += [(None)]
					continue
				for octa in random.sample(octaves,1):
					for amp in random.sample(amplitudes, 1):
						for harm in random.sample(harmonic, 1):
							for hAmp in random.sample(amplitudes, 1):
								lSeqToSound += [(ind,octa,amp,harm,hAmp)] 
		for r_inds in random.sample(partition_ind, 1):
			for r_ind in r_inds:
				if r_ind == 12:
					rSeqToSound += [(None)]
					continue
				for r_octa in random.sample(octaves,1):
					for r_amp in random.sample(amplitudes, 1):
						for r_harm in random.sample(harmonic, 1):
							for r_hAmp in random.sample(amplitudes, 1):
								rSeqToSound += [(r_ind,r_octa,r_amp,r_harm,r_hAmp)]
		num_node += max(len(lSeqToSound),len(rSeqToSound)) #maps redundantly with min or over-unique with max
		l_r_durToSound += [(lSeqToSound, rSeqToSound, dur)]
	return (num_node, l_r_durToSound)

if __name__ == '__main__':
	indices = list(range(13)) #None, rest is 12 #static -> move to global?
	octaves = list(range(1,7))
	durations = [math.pow(1.61804,x)*64 for x in range(1,6)] #[math.pow(1.61804,x)*64 for x in range(8)]
	amplitudes = [math.pow(1.618,x)*0.08 for x in range(2,6)]
	
	#make num_harm harmonics in range
	harmonic = []
	for x in range(1,4):
		for y in range(1,4):
			harmonic += [Fraction(x,y)]
			#make 9 fraction list
	
	
	random.shuffle(indices)
	partition = [3,3,7] # sum 13
	partition_ind = []
	cumul = 0
	for part in partition:
		partition_ind += [indices[cumul:cumul+part]]
		cumul += part
	#partition_notes is like [[0,1,2,3], [4,5,6,7], [8,9,10,11,12], ...]
	#random.shuffle(notes) -> partition_note would be
	# [[10,3,5,4], [6,7,14,12], ...]
	#make list [(combination)] index=linear number
	
	#randomized sampling to reduce space of exploration and # calculations
	
	num_node, l_r_dur = seqToSound(partition_ind,octaves,durations,amplitudes,harmonic)
	print (num_node)
	print (l_r_dur)
    
	seq = makeSeq(scaleFree(num_node),depth=3)
	l, r = makeSoundSeq(seq, l_r_dur)
	#flips and such
	SoundSeqToWav(l,r)
	print ('Check wave file')
