# MinHash and MultMinHash in CSE 312 Assignment 6

## **Introduction**
This assignment focuses on estimating the number of distinct elements in a data stream using **MinHash** and **MultMinHash** techniques. Traditional methods for counting distinct elements require storing all unique values, which is impractical for large-scale data. MinHash provides a memory-efficient alternative by using hash functions and statistical estimation.

## **MinHash**
### **Idea**
MinHash is based on the principle that if we hash elements from a data stream into the range \([0,1]\) using a uniform hash function and take the **minimum** of these hash values, we can estimate the number of distinct elements.

### **Mathematical Basis**
Given a set of \( n \) distinct elements, their hashed values \( U_1, U_2, ..., U_n \) are independent and uniformly distributed over \((0,1)\). The expected minimum value of these hashed values follows:

\[
E[X] = \frac{1}{n+1}
\]

Thus, we can estimate \( n \) using:

\[
n \approx \frac{1}{X} - 1
\]

where \( X \) is the observed minimum hash value.

### **Implementation**
- Each element in the stream is hashed using **MurmurHash** (`mmh3`).
- The minimum hash value seen so far is stored in `self.val`.
- When a new element arrives, if its hash value is smaller than `self.val`, we update `self.val`.
- The estimate of the number of distinct elements is computed using \( \frac{1}{X} - 1 \).

## **MultMinHash**
### **Idea**
A single MinHash instance can have high variance, leading to inaccurate estimates. **MultMinHash** reduces this variance by maintaining multiple independent MinHash estimators with different hash functions (achieved using different `seed_offset` values).

### **Implementation**
- We initialize `num_reps` MinHash objects, each with a unique `seed_offset` to ensure different hash functions.
- Each MinHash object processes the **entire stream**, computing its own minimum hash value.
- The final estimate is computed as:

\[
\bar{X} = \frac{1}{num\_reps} \sum_{i=1}^{num\_reps} X_i
\]

\[
n \approx \frac{1}{\bar{X}} - 1
\]

where \( \bar{X} \) is the average of the minimum hash values across all MinHash instances.

## **Conclusion**
MinHash is an efficient way to estimate distinct elements in a data stream, and **MultMinHash improves accuracy** by averaging multiple independent MinHash estimates. This technique is widely used in **big data processing, search engines, and data deduplication**.

