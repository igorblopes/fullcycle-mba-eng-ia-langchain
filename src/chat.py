from search import search_prompt

def main():
    pergunta = input("Faça uma pergunta: ")
    chain = search_prompt(pergunta)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    print("RESPOSTA: "+chain)
    
    pass

if __name__ == "__main__":
    main()