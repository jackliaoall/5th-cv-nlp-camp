from neo4j import GraphDatabase
import pandas as pd

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "tangyudiadid0"))

k = 10 # nearest neighbors (most similar users) to consider
movies_common = 3 # how many movies in common to be consider an user similar
users_common = 2 # minimum number of similar users that have seen the movie to consider it
threshold_sim = 0.9 # threshold to consider users similar

def load_data():
    with driver.session() as session:
        session.run("""MATCH ()-[r]->() DELETE r""")
        session.run("""MATCH (r) DELETE r""")
        
        print("Loading movies...")
        #加载数据，创建Movie标签,title属性的实体
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_movies.csv" AS csv
            CREATE (:Movie {title: csv.title})
            """)
            
        print("Loading gradings...")
        #加载评分数据，    MERGE是搜索给定模式，如果存在，则返回结果如果它不存在于图中，则它创建新的节点/关系并返回结果。
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_grade.csv" AS csv
            MERGE (m:Movie {title: csv.title}) 
            MERGE (u:User {id: toInteger(csv.user_id)})
            CREATE (u)-[:RATED {grading : toInteger(csv.grade)}]->(m)
            """)
        #加载影片类型数据    
        print("Loading genres...")
            
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_genre.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (g:Genre {genre: csv.genre})
            CREATE (m)-[:HAS_GENRE]->(g)
            """)
            
        print("Loading keywords...")
        #加载关键词数据    
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_keyword.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (k:Keyword {keyword: csv.keyword})
            CREATE (m)-[:HAS_KEYWORD]->(k)
            """)
            
        print("Loading productors...")
        #制片人    
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_productor.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (p:Productor {name: csv.productor})
            CREATE (m)-[:HAS_PRODUCTOR]->(p)
            """)

def queries():
    while True:
        userid = int(input("请输入要为哪位用户推荐电影，输入其ID即可: "))
        m = int(input("为该用户推荐多少个电影呢？ "))
        
        genres = []
        if int(input("是否需要过滤掉不喜欢的类型?（输入0或1）")):#过滤掉不喜欢的类型
            with driver.session() as session:
                try:
                    q = session.run(f"""MATCH (g:Genre) RETURN g.genre AS genre""")
                    result = []
                    for i, r in enumerate(q):
                        result.append(r["genre"])#找到图谱中所有的电影类型
                    df = pd.DataFrame(result, columns=["genre"])
                    print()
                    print(df)
                    inp = input("输入不喜欢的类型索引即可，例如：1 2 3  ")
                    if len(inp) != 0:
                        inp = inp.split(" ")
                        genres = [df["genre"].iloc[int(x)] for x in inp]
                except:
                    print("Error")
                    
        with driver.session() as session:#找到当前ID评分的电影
            q = session.run(f"""
                    MATCH (u1:User {{id : {userid}}})-[r:RATED]-(m:Movie)
                    RETURN m.title AS title, r.grading AS grade
                    ORDER BY grade DESC
                    """)
            
            print()
            print("Your ratings are the following:")
            
            result = []
            for r in q:
                result.append([r["title"], r["grade"]])
                
            if len(result) == 0:
                print("No ratings found")
            else:
                df = pd.DataFrame(result, columns=["title", "grade"])
                print()
                print(df.to_string(index=False))
            print()
            
            session.run(f"""
                MATCH (u1:User)-[s:SIMILARITY]-(u2:User)
                DELETE s
                """)
            #找到当前用户评分的电影以及这些电影被其他用户评分的用户，with是把查询集合当做结果以便后面用where 余弦相似度计算
            session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[r1:RATED]-(m:Movie)-[r2:RATED]-(u2:User)
                WITH
                    u1, u2,
                    COUNT(m) AS movies_common,
                    SUM(r1.grading * r2.grading)/(SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
                WHERE movies_common >= {movies_common} AND sim > {threshold_sim}
                MERGE (u1)-[s:SIMILARITY]-(u2)
                SET s.sim = sim
                """)
                
            Q_GENRE = ""
            if (len(genres) > 0):
                Q_GENRE = "AND ((SIZE(gen) > 0) AND "
                Q_GENRE += "(ANY(x IN " + str(genres) + " WHERE x IN gen))"
                Q_GENRE += ")"
            #找到相似的用户，然后看他们喜欢什么电影 Collect：将所有值收集到一个集合list中
            q = session.run(f"""
                    MATCH (u1:User {{id : {userid}}})-[s:SIMILARITY]-(u2:User)
                    WITH u1, u2, s
                    ORDER BY s.sim DESC LIMIT {k}
                    MATCH (m:Movie)-[r:RATED]-(u2)
                    OPTIONAL MATCH (g:Genre)--(m)
                    WITH u1, u2, s, m, r, COLLECT(DISTINCT g.genre) AS gen
                    WHERE NOT((m)-[:RATED]-(u1)) {Q_GENRE}
                    WITH
                        m.title AS title,
                        SUM(r.grading * s.sim)/SUM(s.sim) AS grade,
                        COUNT(u2) AS num,
                        gen
                    WHERE num >= {users_common}
                    RETURN title, grade, num, gen
                    ORDER BY grade DESC, num DESC
                    LIMIT {m}
                    """)

            print("Recommended movies:")

            result = []
            for r in q:
                result.append([r["title"], r["grade"], r["num"], r["gen"]])
            if len(result) == 0:
                print("No recommendations found")
                print()
                continue
            df = pd.DataFrame(result, columns=["title", "avg grade", "num recommenders", "genres"])
            print()
            print(df.to_string(index=False))
            print()

if __name__ == "__main__":
    if int(input("是否需要重新加载并创建知识图谱？（请选择输入0或1）")):
        load_data()
    queries()
