import unittest

from mymod import plus1, myclass
import mypack

class TestPlus1(unittest.TestCase):

	def test_plus1(self):
		# can hardcode tests
		self.assertEqual(plus1(1), 2)

		# can generate tests in a loop
		for i in range(5):
			self.assertEqual(plus1(i), i + 1)


class TestCube(unittest.TestCase):

	def test_cube(self):
		# can generate tests in a loop
		for i in range(5):
			self.assertEqual(mypack.functions.cube(i), i**3)


class TestMyClass(unittest.TestCase):

	def setUp(self):
		self.a = 5
		self.b = 4
		self.obj = myclass(self.a, self.b)

	def test_a(self):
		self.assertEqual(self.obj.a, self.a)

	def test_b(self):
		# self.assertEqual(self.obj.b, self.a)
		self.assertEqual(self.obj.b, self.b)
