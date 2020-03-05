## LeetCode每日一题高频面试算法题目

### day01 

<img src="images/day01_1.png" style="zoom:80%;" />

```java
class MyStack {
    private Queue<Integer> q1; //输入
    private Queue<Integer> q2; //输出
    private int top;
    /** Initialize your data structure here. */
    public MyStack() {
        q1 = new LinkedList<Integer>();
        q2 = new LinkedList<Integer>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        q1.offer(x); //q1接收数据，将q2中所有数据复制到q1，保证先来的数据在后面，新来的数据在前面
        while(!q2.isEmpty()){
            q1.offer(q2.poll());
        }
        Queue temp = q1;
        q1 = q2;
        q2 = temp;
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        return q2.poll();
    }
    
    /** Get the top element. */
    public int top() {
        return q2.peek();
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return q2.isEmpty();
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```



### day02

<img src="images/day02_1.png" style="zoom: 80%;" />

```java
非递归
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next==null)
            return head;
        ListNode curNode = head.next;
        ListNode preNode = head;
        preNode.next = null;
        ListNode curNext;
        while(curNode!=null){
            curNext = curNode.next;
            curNode.next = preNode;
            preNode = curNode;
            curNode = curNext;
        }
        return preNode;
    }
}
时间复杂度：O(n)，假设 nn 是列表的长度，时间复杂度是 O(n)。
空间复杂度：O(1)。
```

<img src="images/day02_2.png" style="zoom:80%;" />

```java
//1->2->3->4->5:递归执行完向下走的时候，第一次的p指向5，head指向4,head.next是5，当执行head.next.next=head时，p.next指向4，当执行head.next=null时，断开head的4到5的节点完成一次反转，以此类推
public ListNode reverseList(ListNode head){
    if(head==null || head.next==null){
        return head;
    }
    ListNode p = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return p;
}
时间复杂度：O(n)，假设 nn 是列表的长度，那么时间复杂度为 O(n)。
空间复杂度：O(n)，由于使用递归，将会使用隐式栈空间。递归深度可能会达到 n层。
```

### day03

<img src="images/day03_1.png" style="zoom:80%;" />

解析：

<img src="images/day03_2.png" style="zoom: 67%;" />

```java
class Solution {
    public void merge(int[] A, int m, int[] B, int n) {
        int pa = 0;
        int pb = 0;
        int[] sorted = new int[m+n];
        int curVal = 0;
        while(pa<m || pb<n){
            if(pa==m){            //注意判断其中一个是否到达尾部
                curVal = B[pb++];
            }else if(pb==n){
                curVal = A[pa++];
            }else if(A[pa]<B[pb]){
                curVal = A[pa++];
            }else{
                curVal = B[pb++];
            }
           sorted[pa+pb-1] = curVal;
        }
        for(int i=0;i<m+n;i++){
            A[i] = sorted[i];
        }
    }
}
```

### day04

<img src="images/day04_1.png" style="zoom: 80%;" />

1、先将所有腐烂橘子放入Queue（LinkedList）中，建立Map  key=r*C+c  value=此时此刻腐烂橘子所经历的时间

2、当queue不为空 循环遍历，queue  remove得到腐烂橘子队列中的位置，分析该腐烂橘子上下左右使其腐烂，并把腐烂橘子（key=r*C+c, value=上层腐烂橘子对应时间+1）

3、遍历网格，如果有位置为1，说明有橘子未腐烂，return -1，否则返回map中的最大value

```java
class Solution {
    //对行和列进行移动，上，左，下，右
    int[] dr = new int[]{-1,0,1,0};
    int[] dc = new int[]{0,-1,0,1};
    public int orangesRotting(int[][] grid) {
        int R = grid.length;
        int C = grid[0].length;
        
        Queue<Integer> queue = new LinkedList();
        Map<Integer,Integer> depth = new HashMap<>();
        //先遍历寻找该开始就腐烂的橘子
        for(int r=0;r<R;r++){
            for(int c=0;c<C;c++){
                if(grid[r][c]==2){
                    int code = r*C+c; //将表格中腐烂橘子的二维坐标转化为一个数字编码
                    queue.add(code);
                    depth.put(code,0); //key为二维坐标对应的数字编码，value为该编码对应的橘子腐烂用时
                }
            }
        }
        
        int ans = 0;
        while(!queue.isEmpty()){
            int code = queue.remove();
            int r = code/C;
            int c = code%C;
            for(int k=0;k<4;k++){  //将该腐烂橘子的上下左右依次腐烂
                int nr = r + dr[k];
                int nc = c + dc[k];
                if(nr>=0 && nr<R && nc>=0 && nc<C && grid[nr][nc]==1){
                    grid[nr][nc] = 2;
                    int ncode = nr*C+nc;
                    queue.add(ncode);
                    depth.put(ncode,depth.get(code)+1); //对腐烂橘子的时刻进行重新设定  注意depth.get(code)不是ncode 
                    ans = depth.get(ncode);
                }
            }
        }
        
        for(int[] r:grid){
            for(int c :r){
                if(c==1){
                    return -1;
                }
            }
        }
        return ans;
        
    }
}

时间复杂度：O(nm)
即进行一次广度优先搜索的时间，其中 n=grid.lengthn=grid.length, m=grid[0].lengthm=grid[0].length 

空间复杂度：O(nm)
需要额外的 disdis 数组记录每个新鲜橘子被腐烂的最短时间，大小为 O(nm)，且广度优先搜索中队列里存放的状态最多不会超过 nmnm 个，最多需要 O(nm) 的空间，所以最后的空间复杂度为 O(nm)。

```

### day05

<img src="images/day05_1.png" style="zoom:80%;" />

```java
class Solution {
    public int[] distributeCandies(int candies, int num_people) {
        int[] ans = new int[num_people];
        int i = 0;
        while(candies!=0){
            ans[i%num_people] += Math.min(candies,i+1);  //这里i%num_people用来多次循环数组
            candies -= Math.min(candies,i+1);
            i +=1;
        }
        return ans;
    }
}
```

