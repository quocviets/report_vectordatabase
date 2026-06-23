# RAG Retrieval Optimization Research Note

## Tổng quan Workflow

### Query Processing

```text
Query
 ↓
Embedding Model
 ↓
Query Vector
 ↓
FAISS Similarity Search
 ↓
Calculate Similarity Score
 ↓
Ranking Result
 ↓
Return Top-K Relevant Chunks
```

### Database Processing

```text
PDF
 ↓
Text Extraction
 ↓
Data Cleaning
 ↓
Chunking Strategy
 ↓
Embedding Model
 ↓
Vector Embedding
 ↓
FAISS Index
 ↓
Semantic Search Database
```

---

# 1. Chunking Strategy

## 1.1 Structure-based Chunking (Done)

### Ý tưởng

Một Disease sẽ được xem là một Chunk.

Ví dụ:

```text
Bệnh Đạo ôn
├── Cơ chế và Dấu hiệu
├── Nguyên nhân
└── Cách điều trị

=> 1 Disease = 1 Chunk
```

### Workflow

```text
PDF Structure
 ↓
Disease Detection
 ↓
Disease Chunking
 ↓
Embedding Model
 ↓
Vector Embedding
 ↓
FAISS Index
 ↓
Semantic Search
```

### Ưu điểm

- Cấu trúc đơn giản.
- Ít số lượng Vector.
- Tốc độ tạo Index và Search nhanh.

### Nhược điểm

Chunk chứa quá nhiều thông tin.

Ví dụ:

Query:
```
Cách điều trị bệnh Đạo ôn
```

Kết quả:
```
Bệnh Đạo ôn
- Cơ chế và Dấu hiệu
- Nguyên nhân
- Cách điều trị
```

Người dùng vẫn cần đọc toàn bộ nội dung để tìm phần mong muốn.

---

## 1.2 Section-based Chunking (Done)

### Ý tưởng

Thay vì lưu toàn bộ thông tin của một Disease vào một Chunk.

Chia nhỏ theo từng Section:

```text
1 Disease + 1 Section = 1 Chunk
```

Ví dụ:

```text
Bệnh Đạo ôn

Chunk 1:
Disease: Đạo ôn
Section: Cơ chế và Dấu hiệu

Chunk 2:
Disease: Đạo ôn
Section: Nguyên nhân

Chunk 3:
Disease: Đạo ôn
Section: Cách điều trị
```

### Workflow

```text
PDF
 ↓
Text Extraction
 ↓
Data Cleaning
 ↓
Disease Detection
 ↓
Section Detection (Rule-based)
 ↓
Section-based Chunking
 ↓
Add Metadata (Optional)
 ↓
Embedding Model
 ↓
Vector Embedding
 ↓
FAISS Index
 ↓
Semantic Search
```

### Vấn đề gặp phải

#### Vấn đề 1: Fixed-size Chunking làm mất ngữ cảnh

Nguyên nhân:

- Chia theo số lượng ký tự hoặc số lượng token.
- Không hiểu đâu là đầu câu hoặc cuối câu.
- Có thể cắt mất thông tin quan trọng.

Ví dụ:

```text
Cây lúa bị bệnh...
Cách điều trị sử dụng thuốc Tricy...
```

Nội dung bị cắt giữa chừng.

Solution:

Chuyển sang Structure-based Chunking.

Kết quả:

- Giữ nguyên được toàn bộ thông tin của một Disease.
- Không còn hiện tượng mất phần đầu hoặc phần cuối của nội dung.

---

#### Vấn đề 2: Trả về thông tin quá rộng

Ví dụ:

Query:

```text
Cách điều trị bệnh Đạo ôn
```

Structure-based Chunking:

```text
Bệnh Đạo ôn
- Cơ chế và dấu hiệu
- Nguyên nhân
- Cách điều trị
```

Mặc dù đúng Disease nhưng chưa trả về đúng thông tin mà người dùng cần.

Solution:

Sử dụng Section-based Chunking:

```text
Disease + Section = 1 Chunk
```

Kết quả mong muốn:

```text
Disease: Đạo ôn
Section: Cách điều trị
Content: Sử dụng Tricyclazole...
```

### Ưu điểm

- Trả về thông tin chi tiết hơn.
- Tăng độ chính xác khi Question Answering.
- Giảm lượng context không cần thiết đưa vào LLM.

### Nhược điểm

- Số lượng Chunk tăng lên.
- Số lượng Vector Embedding tăng lên.
- Tăng thời gian Indexing và dung lượng lưu trữ.

---

# 2. Similarity Metric Comparison

Sau khi hoàn thành Section-based Chunking, tiến hành đánh giá các Similarity Metric trong Semantic Search.

Mục tiêu:

```text
Query
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Similarity Calculation
 ↓
Ranking
 ↓
Top-K Relevant Chunks
```

Thực hiện so sánh:

- L2 Distance
- Cosine Similarity

---

## 2.1 L2 Distance

### Khái niệm

L2 Distance (Euclidean Distance) tính khoảng cách giữa hai Vector trong Vector Space.

Công thức:

```text
Distance(A, B) = ||A - B||
```

Ý nghĩa:

```text
Distance nhỏ hơn
        ↓
Hai Vector gần nhau hơn trong Vector Space
```

Lưu ý:

Khoảng cách nhỏ không đồng nghĩa với việc kết quả chính xác hơn theo góc nhìn của con người.

Nó chỉ thể hiện rằng hai Vector gần nhau trong Embedding Space.

### Workflow

```text
Query
 ↓
Embedding Model
 ↓
Query Vector
 ↓
FAISS IndexFlatL2
 ↓
Calculate L2 Distance
 ↓
Ranking
 ↓
Return Top-K
```

---

## 2.2 Cosine Similarity

### Khái niệm

Cosine Similarity đo góc giữa hai Vector.

Đặc điểm:

- Chỉ quan tâm đến hướng của Vector.
- Không quan tâm đến độ dài của Vector.

Ý nghĩa:

```text
Góc nhỏ
 ↓
Similarity cao
 ↓
Ngữ nghĩa gần nhau hơn
```

### Workflow

```text
Query
 ↓
Embedding Model
 ↓
Normalize Vector
 ↓
FAISS IndexFlatIP
 ↓
Calculate Cosine Similarity
 ↓
Ranking
 ↓
Return Top-K
```

### Kết quả thực nghiệm ban đầu

Query:

```text
Cách điều trị bệnh Đạo ôn
```

L2 và Cosine đều trả về:

```text
Top 1: Bệnh Tuyến trùng (Sai)
Top 2: Bệnh Đạo ôn - Cách điều trị (Đúng)
```

Nguyên nhân:

Semantic Search hiểu được các từ:

```text
điều trị
phác đồ
thuốc
```

Nhưng chưa ưu tiên đủ Keyword quan trọng:

```text
Đạo ôn
```

Kết luận:

- L2 và Cosine có kết quả tương đối giống nhau trên Dataset hiện tại.
- Semantic Search có khả năng hiểu ngữ nghĩa nhưng có thể bỏ sót Exact Keyword.
- Cần kết hợp thêm Keyword-based Retrieval để cải thiện độ chính xác.
# 3. BM25 Keyword Search

## 3.1 Khái niệm

BM25 (Best Matching 25) là một thuật toán Keyword-based Retrieval.

Khác với Semantic Search sử dụng Vector Embedding để hiểu ngữ nghĩa, BM25 sẽ đánh giá mức độ liên quan dựa trên sự xuất hiện của các từ khóa trong Document.

Ví dụ:

Query:

```text
Cách điều trị bệnh Đạo ôn
```

BM25 sẽ tập trung vào các Keyword:

```text
Cách
điều trị
bệnh
Đạo ôn
```

Document chứa nhiều Keyword quan trọng sẽ có BM25 Score cao hơn.

---

## 3.2 Workflow

```text
Document
 ↓
Tokenization
 ↓
BM25 Index
 ↓
User Query
 ↓
Query Tokenization
 ↓
Calculate BM25 Score
 ↓
Ranking
 ↓
Return Top-K Documents
```

---

## 3.3 Cách hoạt động

BM25 tính điểm dựa trên 3 yếu tố chính:

### Term Frequency (TF)

Đánh giá số lần xuất hiện của một Keyword trong Document.

Ví dụ:

```text
Document A:
Bệnh Đạo ôn gây cháy lá.
Đạo ôn phát triển mạnh trong điều kiện độ ẩm cao.
```

Keyword:

```text
Đạo ôn
```

Xuất hiện nhiều lần → TF cao → Score tăng.

---

### Inverse Document Frequency (IDF)

Đánh giá độ hiếm của Keyword trong toàn bộ Dataset.

Ví dụ:

Keyword:

```text
bệnh
```

Xuất hiện trong hầu hết Document.

→ IDF thấp.

Keyword:

```text
Tricyclazole
```

Chỉ xuất hiện trong một vài Document.

→ IDF cao → Có giá trị phân biệt lớn.

---

### Document Length Normalization

Điều chỉnh Score dựa trên độ dài của Document.

Ví dụ:

```text
Document A:
100 từ chứa "Đạo ôn"

Document B:
1000 từ chứa "Đạo ôn"
```

Document A có thể được đánh giá cao hơn vì thông tin tập trung hơn.

---

## 3.4 Ưu điểm

- Rất tốt với Exact Keyword.
- Hiệu quả với tên bệnh, tên thuốc, tác nhân gây bệnh.
- Không cần Embedding Model.
- Tốc độ Index nhanh.

---

## 3.5 Nhược điểm

Không hiểu ngữ nghĩa.

Ví dụ:

Query:

```text
Lúa bị cháy lá xử lý thế nào?
```

Document:

```text
Bệnh Đạo ôn gây cháy lá, sử dụng thuốc Tricyclazole để điều trị.
```

BM25 có thể bỏ sót vì không hiểu:

```text
xử lý ≈ điều trị
```

hoặc các cách diễn đạt tương đương.

---

## 3.6 Kết quả thực nghiệm

Query:

```text
Cách điều trị bệnh Đạo ôn
```

Kết quả:

```text
Top 1:
Bệnh Đạo ôn - Cách điều trị

Top 2:
Bệnh Đạo ôn - Cơ chế và Dấu hiệu

Top 3:
Bệnh Đạo ôn - Nguyên nhân
```

Phân tích:

BM25 ưu tiên Keyword:

```text
Đạo ôn
Cách điều trị
```

nên có khả năng trả về đúng Disease và đúng Section.

---

## Kết luận

BM25 hoạt động tốt trong các trường hợp có Keyword cụ thể như:

- Tên bệnh.
- Tên thuốc.
- Tên tác nhân gây bệnh.

Tuy nhiên, BM25 không hiểu được ngữ nghĩa của câu hỏi tự nhiên.

Do đó cần kết hợp với Semantic Search để tận dụng ưu điểm của cả hai phương pháp.

---

# 4. Hybrid Search (BM25 + Semantic Search)

## 4.1 Mục tiêu

Kết hợp:

- BM25 giữ lại Exact Keyword.
- Semantic Search hiểu ý nghĩa của Query.

Mục tiêu là cải thiện độ chính xác của quá trình Retrieval.

---

## 4.2 Workflow

```text
                     Query
                       |
          --------------------------
          |                        |
          ↓                        ↓
    BM25 Keyword Search       Embedding Model
          ↓                        ↓
      BM25 Score             Query Vector
                                   ↓
                             FAISS Cosine Search
                                   ↓
                            Cosine Similarity Score
          |                        |
          -------- Score Normalization --------
                          |
                          ↓
                   Score Combination
                          |
                          ↓
                        Ranking
                          |
                          ↓
                 Return Top-K Chunks
```

---

## 4.3 Score Combination

Do BM25 và Cosine Similarity sử dụng hai thang điểm khác nhau:

Ví dụ:

```text
BM25 Score:
9.26

Cosine Score:
0.56
```

Không thể cộng trực tiếp:

```text
9.26 + 0.56 ❌
```

Cần Normalize về cùng một khoảng giá trị:

```text
0 → 1
```

Sau đó tính Hybrid Score:

```text
Hybrid Score =
α × BM25 Score +
(1 - α) × Cosine Score
```

Trong thực nghiệm:

```text
α = 0.5
```

Tức là:

- 50% Keyword Matching.
- 50% Semantic Similarity.

---

## 4.4 Kết quả thực nghiệm

### Case 1

Query:

```text
Cách điều trị bệnh Đạo ôn
```

Cosine Search:

```text
Top 1:
Bệnh Tuyến trùng (Sai)

Nguyên nhân:
Semantic hiểu "điều trị", "phác đồ", "thuốc"
nhưng không ưu tiên đủ Keyword "Đạo ôn".
```

BM25:

```text
Top 1:
Bệnh Đạo ôn - Cách điều trị (Đúng)
```

Hybrid Search:

```text
Top 1:
Bệnh Đạo ôn - Cách điều trị (Đúng)
```

---

### Case 2

Query:

```text
Cách điều trị bệnh Tuyến trùng
```

Hybrid Search:

```text
Top 1:
Bệnh Tuyến trùng - Phác đồ điều trị
```

Phân tích:

Cosine nhận biết sự tương đồng giữa:

```text
Cách điều trị ≈ Phác đồ điều trị
```

BM25 đảm bảo Keyword:

```text
Tuyến trùng
```

được ưu tiên.

---

## 4.5 Kết luận

Hybrid Search tận dụng được ưu điểm của cả hai phương pháp:

### BM25

Mạnh về:

- Exact Keyword.
- Tên bệnh.
- Tên thuốc.
- Thuật ngữ chuyên ngành.

---

### Semantic Search

Mạnh về:

- Hiểu ngữ nghĩa.
- Hiểu các cách diễn đạt khác nhau.
- Tìm kiếm thông tin tương đồng.

---

### Hybrid Search

Kết hợp:

```text
Keyword Accuracy
        +
Semantic Understanding
        =
Better Retrieval Performance
```

Kết quả thực nghiệm cho thấy Hybrid Search có khả năng trả về kết quả chính xác hơn so với việc chỉ sử dụng BM25 hoặc Semantic Search riêng lẻ.

---

# 6. Evaluation (Completed)

## 6.1 Evaluation Method

Sử dụng tập dữ liệu kiểm thử gồm 20 Query liên quan đến:

- Tên Disease.
- Symptoms / Dấu hiệu.
- Cause / Nguyên nhân.
- Treatment / Cách điều trị.
- Tên thuốc và tác nhân gây bệnh.

Workflow:

```text
Test Query
     ↓
Retrieval Method
(BM25 / Cosine / Hybrid)
     ↓
Return Top-K Chunks
     ↓
Compare với Ground Truth
     ↓
Calculate Top-1 và Top-3 Accuracy
```

---

## 6.2 Evaluation Result

| Method | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|
| Cosine Similarity | 55% | 85% |
| BM25 Keyword Search | 75% | 90% |
| Hybrid Search | **85%** | **95%** |

---

## 6.3 Analysis

### Cosine Similarity

Ưu điểm:

- Hiểu được Semantic Meaning.
- Có khả năng nhận biết các cách diễn đạt tương tự.

Ví dụ:

```text
Cách điều trị ≈ Phác đồ điều trị
```

Nhược điểm:

- Không ưu tiên mạnh Exact Keyword.
- Có thể trả về Disease khác nhưng có nội dung tương tự.

Ví dụ:

```text
Query:
Cách điều trị bệnh Đạo ôn

Result:
Bệnh Vàng lùn hoặc Tuyến trùng
```

---

### BM25 Keyword Search

Ưu điểm:

- Hoạt động tốt với:
  - Tên Disease.
  - Tên thuốc.
  - Tác nhân gây bệnh.
  - Thuật ngữ chuyên ngành.

Nhược điểm:

Không hiểu được ngữ nghĩa.

Ví dụ:

```text
Query:
Cách xử lý bệnh Lép vàng

Document:
Phác đồ điều trị bệnh Lép vàng
```

BM25 không hiểu:

```text
Xử lý ≈ Điều trị
```

---

### Hybrid Search

Kết hợp:

```text
BM25 + Semantic Search
```

Giúp tận dụng:

- Keyword Matching từ BM25.
- Semantic Understanding từ Embedding Model.

Kết quả thực nghiệm cho thấy Hybrid đạt độ chính xác cao nhất.

---

# 7. Error Analysis

Sau khi đánh giá Hybrid Search, vẫn còn một số trường hợp Retrieval chưa chính xác.

## Case 1: Vietnamese Keyword Splitting

Ví dụ:

Query:

```text
Dấu hiệu bệnh Bạc lá
```

Hiện tại Tokenization:

```text
Bạc lá
↓
bạc + lá
```

Điều này làm mất ý nghĩa của cụm từ chuyên ngành:

```text
Bạc lá
Đạo ôn
Lép vàng
```

Nguyên nhân:

Hiện tại BM25 sử dụng:

```python
text.lower().split()
```

Chỉ tách theo khoảng trắng, không phù hợp với tiếng Việt.

---

## Case 2: Semantic Search quá tổng quát

Ví dụ:

```text
Cách xử lý bệnh Lép vàng
```

Cosine có thể trả về:

```text
Lúa cỏ
```

Do các Document cùng chứa các khái niệm:

```text
Xử lý
Điều trị
Thuốc
```

nhưng không phải cùng Disease.

---

## Case 3: Tên thuốc chuyên biệt

Ví dụ:

```text
Thuốc Tervigo dùng cho bệnh gì?
```

Expected:

```text
Tuyến trùng
```

Tuy nhiên Hybrid vẫn có thể ưu tiên Disease khác.

Cần kiểm tra lại:
- BM25 Score.
- Cosine Score.
- Cách kết hợp Hybrid Score.

---

# 8. Current Limitation

- BM25 sử dụng whitespace Tokenization.
- Chưa tối ưu cho từ ghép tiếng Việt.
- Embedding Model hiện tại là general multilingual model.
- Hybrid Score đang sử dụng alpha = 0.5, chưa thực hiện tuning.

---

# 9. Future Improvement

## 9.1 Vietnamese Word Segmentation

Thay thế:

```python
text.lower().split()
```

bằng Vietnamese Tokenizer:

- Underthesea.
- PyVi.
- VnCoreNLP.

Ví dụ:

Trước:

```text
Bệnh bạc lá
↓
bệnh + bạc + lá
```

Sau:

```text
Bệnh bạc_lá
```

Mục tiêu:

- Tăng độ chính xác của BM25.
- Cải thiện Hybrid Search.

---

## 9.2 Hybrid Parameter Tuning

Hiện tại:

```text
Hybrid Score = 0.5 × BM25 + 0.5 × Cosine
```

Có thể thử:

```text
alpha = 0.6
alpha = 0.7
alpha = 0.8
```

để tăng ảnh hưởng của Keyword Matching.

---

## 9.3 Compare Embedding Model

Model hiện tại:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Các hướng thử nghiệm tiếp theo:

- BGE-M3.
- multilingual-e5-small.
- PhoBERT (nếu phù hợp).

---

## Current Progress

```text
PDF Processing                     ✅

Data Cleaning                      ✅

Chunking Strategy
- Structure-based Chunking         ✅
- Section-based Chunking           ✅

Semantic Search
- L2 Distance                      ✅
- Cosine Similarity                ✅

Keyword Search
- BM25                             ✅

Hybrid Search                      ✅

Evaluation
- Top-1 Accuracy                   ✅
- Top-3 Accuracy                   ✅

Error Analysis                     ✅


