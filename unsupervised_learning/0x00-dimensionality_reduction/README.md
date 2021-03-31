# dimensionality reduction


pricipal component analysis and single value decomposition

data:
curl -o mnist2500_labels.txt 'https://holbertonintranet.s3.amazonaws.com
/uploads/text/2019/10/72a86270e2a1c2cbc14b.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Crede
ntial=AKIARDDGGGOUWMNL5ANN%2F20210324%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210324T1
61801Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=0b53e44fc08f7f86bea6ba
2e8bbc428bfb512f9ce2f53b7d3c4af1a3f262d9f1'

curl -o mnist2500_X.txt 'https://intranet-projects-files.s3.amazonaws.co
m/holbertonschool-ml/mnist2500_X.txt'

i believe the first was too large for github


additional sources:
for 0 and 1 this helped explain the variance thing:
https://stats.stackexchange.com/questions/132886/how-many-components-to-use-in-pca-in-order-to-preserve-a-certain-amount-of-varia
for 1 make sure to read the main file of 0 closely

this is basic but helps with what svd returns if you don't know:
https://www.google.com/search?q=np.linalg.svd&source=hp&ei=lYBbYKSsMYy9kgXf5oTwCg&iflsig=AINFCbYAAAAAYFuOpZpeMlGc9V3bSeAgCHZzCYc6XosQ&oq=np.linalg.svd&gs_lcp=Cgdnd3Mtd2l6EAMyAggAMgIIADICCAAyAggAMgIIADoFCAAQsQM6AgguOggIABCxAxCDAVBfWLgaYNkjaABwAHgBgAHUBYgBnyCSAQsyLTMuMS40LjEuMZgBAKABAaoBB2d3cy13aXo&sclient=gws-wiz&ved=0ahUKEwjkyre2w8nvAhWMnqQKHV8zAa4Q4dUDCAk&uact=5


