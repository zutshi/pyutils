'''
Implements stream writing of data after pcikling and compressing them.
'''

# TODO: optimization
# https://www.python.org/doc/essays/list2str/

import cPickle as cP

import zlib

import fileops as fops

from IPython import embed


# Problem is, the delmiter is not unique. And the pickled data can
# have characters which mimic DELIM. Hence, does not look lik ethere
# is a way around it.
DELIM = str('\n\n\n\n')


class PickleStreamReader(object):
    """Reads a file writtern by PickleStreamWriter

       read(): data = unpickle(uncompress(fileread))

       Provides a read function which returns a generator of stored
       pickles. Uses zlib for compression and cPickle for pickling
       operations.

    """
    def __init__(self, fname, compression=True):
        self.fname = fname
        return

    def read(self):
        with fops.ReadLine(self.fname, mode='rb') as sr:
            ucp = PickleStreamUnCompressor(sr)
            for data in ucp.uncompressor():
                yield data
        return

    def read_older(self):
        zo = zlib.decompressobj()
        partial_pickle = []
        done = False
        with fops.ReadLine(self.fname, mode='rb') as sr:
            while not done:
                cdata = sr.readline()
                if cdata == str(''):
                    done = True
                    ucdata = zo.flush()
                else:
                    ucdata = zo.decompress(cdata)

                buf = ucdata.split(DELIM)
                assert(len(buf) > 0)

                # if split occurred
                if len(buf) > 1:
                    partial_pickle.append(buf[0])
                    # pickle is completed
                    pickle = str('').join(partial_pickle)
                    yield cP.loads(pickle)
                    partial_pickle = []
                    for pickle in buf[1:-1]:
                        yield cP.loads(pickle)
                    partial_pickle.append(buf[-1])
                else:
                    partial_pickle.append(buf[0])

        assert(partial_pickle == ['', ''])
        return


class PickleStreamWriter(object):
    """Pickles and compresses data and writes it to a file.
        Provides a function

        writer(): filewrite(compress(pickle(data)))

        which pickles and compresses the given data, before writing
        it. Uses cpickle and zlib for pickling and compression, resp.
        The pickling protocol used is HIGHEST_PROTOCOL.
        The default compression level for zlib is used.

        """
    def __init__(self, fname, compression=True):
        self.fname = fname

        self.sw = None
        self.comp = None
        return

    def write(self, data):
        self.sw.write(self.comp.compress(data))
        return

    def __enter__(self):
        self.sw = fops.StreamWrite(self.fname, mode='wb')
        self.comp = PickleStreamCompressor()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sw.write(self.comp.flush())
        self.sw.close_file()
        return


class PickleStreamWriterIter(object):
    """Like PickleStreamWriter, but the writer() takes in an iterator.
       Like a one-shot writer.
    """
    def __init__(self, fname, compression=True):
        self.fname = fname

    def write(self, data):
        zo = zlib.compressobj()
        with fops.StreamWrite(self.fname, mode='wb') as sw:
            for d in data:
                pickled = cP.dumps(d, protocol=cP.HIGHEST_PROTOCOL)
                # write size
                sz = len(pickled)
                # Two newlines around the size make life easy
                sw.write(zo.compress('\n{}\n'.format(sz)))
                sw.write(zo.compress(pickled))
            sw.write(zo.flush())


#     def write(self, trace):
#         zo = zlib.compressobj()
#         with fops.StreamWrite(self.fname, mode='wb') as sw:
#             # pickle the trace and dump it
#             # Remove pickling from here...this should be the lightest
#             # process as it is the bottleneck
#             pickled_trace = cP.dumps(trace, protocol=cP.HIGHEST_PROTOCOL)
#             # add teo newlines, as this is *never*?? happens in a
#             # pickle?
#             #sw.write(DELIM)
#             # write size
#             sz = len(pickled_trace)
#             sw.write(zo.compress('\n{}\n'.format(sz)))
#             sw.write(zo.compress(pickled_trace))
#             #sw.write(zo.compress('\n\n\n\n'))
#             # can serialize, but then will have to do book keeping
#             # for de serializing
#             #sw.write(trace.serialize())
#             sw.write(zo.flush())

#     def trace_gen(self):
#         with fops.ReadLine(self.fname, mode='rb') as sr:
#             data = []
#             line = sr.read()
#             while line != str(''):
#                 if line == str('\n'):
#                     # remove the last '\n'
#                     data[-1] = data[-1][:-1]
#                     yield cP.loads(str('').join(data))
#                     data = []
#                 else:
#                     data.append(line)
#                 line = sr.read()
#         return

class PickleStreamUnCompressor(object):
    """un-compresses a pickled stream from PickleStreamCompressor

       read(): data = return unpickle(uncompress(stream))

       Provides a read function which returns a generator of stored
       pickles. Uses zlib for compression and cPickle for pickling
       operations.

    """
    def __init__(self, stream, compression=True):
        """__init__

        Parameters
        ----------
        stream : stream must support the readline() method:
                 Reads till newline and the returned string has
                 newline at the end. EOF is indicated by ''.
                 This follows the exact convention of python's
                 readline()
        compression : Is on, can not be turned off.
        """
        self.stream = stream
        self.zo = zlib.decompressobj()
        return

    def get_data(self):
        ucdata = ''
        while not ucdata:
            cdata = self.stream.readline()
            if cdata == '':
                return self.zo.flush()
            ucdata = self.zo.decompress(cdata)
        return ucdata

    def uncompressor(self):
        DELIM = '\n'
        MT = ''
        lbuf = MT
        while True:
            while lbuf.count(DELIM) < 2:
                data = self.get_data()
                lbuf += data
                if data == MT:
                    return
            _, l, buf = lbuf.split(DELIM, 2)
            l = int(l)
            assert(_ == MT)
            while len(buf) < l:
                D = self.get_data()
                assert(D != MT)
                buf += D
            pickle, partial_pickle = buf[0:l], buf[l:]
            yield cP.loads(pickle)
            lbuf = partial_pickle

        return
#     def uncompressor(self):
#         notdone = True
#         empty = str('')
#         DELIM = str('\n')
#         while True:
#             ucdata = self.get_data()
#             partial_pickle = empty
#             while notdone:
#                 try:
#                     _, sz, partial_pickle = ucdata.split(DELIM, 2)
#                 except:
#                     embed()
#                 assert(_ == empty)
#                 assert(partial_pickle != empty)
#                 sz = int(sz)
#                 assert(sz > 0)
#                 while len(partial_pickle) < sz:
#                     partial_pickle += self.get_data()#self.zo.decompress(self.stream.readline())
#                 pickle, partial_pickle = partial_pickle[0:sz], partial_pickle[sz:]
#                 yield cP.loads(pickle)
#                 notdone = bool(partial_pickle)
#                 ucdata = str(partial_pickle)
#         return


class PickleStreamCompressor(object):
    """Pickles and compresses data and writes it to a stream.
        Provides a function

        writer(): return compress(pickle(data))

        which pickles and compresses the given data.
        Uses cpickle and zlib for pickling and compression, resp.
        The pickling protocol used is HIGHEST_PROTOCOL.
        The default compression level for zlib is used.

        """
    def __init__(self, compression=True):
        self.zo = zlib.compressobj()
        return

    def compress(self, data):
        pickle = cP.dumps(data, protocol=cP.HIGHEST_PROTOCOL)
        # size of pickle
        sz = len(pickle)
        # compressed pickle
        #cpickle = self.zo.compress(pickle)
        # Two newlines around the size make life easy
        return self.zo.compress('\n{}\n{}'.format(sz, pickle))

    def flush(self):
        """MUST be called at the end to flush remaining data if not
        using as a context manager.
        Follows zlib's interface"""
        return self.zo.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
